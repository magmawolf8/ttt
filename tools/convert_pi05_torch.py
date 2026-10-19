"""Memory-bounded conversion of the retained official OpenPI checkpoint.

Uses the two tensor-layout conversion functions verbatim from the pinned upstream
script, without importing that script's unrelated training dependencies. Weights
are restored in bf16, as in the existing JAX inference path, and saved in shards.
The two unused language-generation heads are excluded from this action-only model.
"""
import ast
import gc
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import torch
from flax.nnx import traversals
from safetensors.torch import save_file

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from openpi.models import gemma
from openpi.models import model as model_module
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


def build_meta_model():
    config=Pi0Config(pi05=True,action_horizon=10,discrete_state_input=False,pytorch_compile_mode=None)
    with torch.device("meta"):
        model=PI0Pytorch(config)
    # sample_actions calls the underlying transformer models, never these heads.
    model.paligemma_with_expert.paligemma.lm_head=torch.nn.Identity()
    model.paligemma_with_expert.gemma_expert.lm_head=torch.nn.Identity()
    return model


def numpy_to_torch(array):
    array=np.asarray(array)
    if array.dtype==np.dtype(ml_dtypes.bfloat16):
        return torch.from_numpy(array.view(np.uint16)).view(torch.bfloat16)
    return torch.from_numpy(array)


def upstream_slicers():
    path=ROOT/"vendor/openpi/examples/convert_jax_model_to_pytorch.py"
    source=path.read_text()
    names={"slice_paligemma_state_dict","slice_gemma_state_dict"}
    nodes=[node for node in ast.parse(source).body if isinstance(node,ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes}!=names:
        raise RuntimeError("Pinned upstream converter functions not found")
    # The upstream helpers only need these two torch symbols. This shim adds
    # NumPy bf16 support without an intermediate fp32 copy of every parameter.
    namespace={"torch":SimpleNamespace(from_numpy=numpy_to_torch,Tensor=torch.Tensor)}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),"exec"),namespace)
    return namespace["slice_paligemma_state_dict"],namespace["slice_gemma_state_dict"],hashlib.sha256(source.encode()).hexdigest()


def main():
    destination=ROOT/"models/pi05_libero_torch"
    destination.mkdir(parents=True,exist_ok=True)
    model=build_meta_model()
    expected=model.state_dict()
    print(f"Inference model: {sum(p.numel() for p in model.parameters()):,} parameters",flush=True)
    print("Restoring original checkpoint as bf16 NumPy arrays...",flush=True)
    params=model_module.restore_params(ROOT/"models/pi05_libero/params",restore_type=np.ndarray,dtype=ml_dtypes.bfloat16)
    print("Checkpoint restored; converting layouts",flush=True)
    slice_paligemma,slice_expert,source_hash=upstream_slicers()
    vision=SimpleNamespace(hidden_size=1152,num_hidden_layers=27,num_attention_heads=16,intermediate_size=4304,patch_size=14,projection_dim=2048)
    text=SimpleNamespace(hidden_size=2048,num_hidden_layers=18,num_attention_heads=8,head_dim=256,intermediate_size=16384)
    base,expert=slice_paligemma(dict(traversals.flatten_mapping(params["PaliGemma"],sep="/")),SimpleNamespace(vision_config=vision,text_config=text))
    converted=slice_expert(expert,gemma.get_config("gemma_300m"),num_expert=1,checkpoint_dir="pi05_libero",pi05=True)
    converted.update(base)
    for name in ["action_in_proj","action_out_proj","time_mlp_in","time_mlp_out"]:
        converted[name+".weight"]=numpy_to_torch(params[name]["kernel"]).T
        converted[name+".bias"]=numpy_to_torch(params[name]["bias"])
    missing=set(expected)-set(converted)
    extra=set(converted)-set(expected)
    if missing or extra:
        raise ValueError(f"Conversion key mismatch: missing={missing}, extra={extra}")
    for key,value in converted.items():
        if value.shape!=expected[key].shape:
            raise ValueError(f"Shape mismatch: {key}: {value.shape} != {expected[key].shape}")
    total_bytes=sum(value.numel()*value.element_size() for value in converted.values())
    print(f"Validated all {len(converted)} tensors, {total_bytes/1e9:.2f} GB",flush=True)
    shards=[]
    shard={}
    shard_bytes=0
    weight_map={}
    def write_shard():
        nonlocal shard,shard_bytes
        if not shard:
            return
        filename=f"model-{len(shards)+1:04d}.safetensors"
        # Contiguity copies are limited to this shard; no second full model copy.
        contiguous={key:value.contiguous() for key,value in shard.items()}
        save_file(contiguous,str(destination/filename))
        weight_map.update({key:filename for key in shard})
        shards.append(filename)
        print(f"Saved {filename}: {shard_bytes/1e6:.1f} MB",flush=True)
        shard={}
        shard_bytes=0
        del contiguous
        gc.collect()
    for key,value in converted.items():
        nbytes=value.numel()*value.element_size()
        if shard_bytes+nbytes>256*1024**2:
            write_shard()
        shard[key]=value
        shard_bytes+=nbytes
    write_shard()
    manifest={"source":"gs://openpi-assets/checkpoints/pi05_libero","source_converter_sha256":source_hash,"precision":"bfloat16","unused_language_heads_removed":True,"tensor_count":len(converted),"total_bytes":total_bytes,"weight_map":weight_map}
    (destination/"model.safetensors.index.json").write_text(json.dumps(manifest,indent=2))
    print(f"Conversion complete: {destination}",flush=True)


if __name__=="__main__":
    main()
