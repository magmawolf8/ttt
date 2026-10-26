"""OpenPI action-only PyTorch loader for the 8 GB RTX 4070."""
import gc
import json
from pathlib import Path
import time

import torch
from safetensors import safe_open

ROOT=Path(__file__).resolve().parent


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use an NVIDIA GPU and the CUDA PyTorch environment from setup_panda_cuda.py.")
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("This checkpoint runtime requires native NVIDIA bfloat16 support (Ampere or newer).")


def load_cuda_model():
    require_cuda()
    from tools.convert_pi05_torch import build_meta_model

    destination=ROOT/"models/pi05_libero_torch"
    index_path=destination/"model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError("Convert weights first: python tools/convert_pi05_torch.py")
    index=json.loads(index_path.read_text())
    model=build_meta_model()
    expected=model.state_dict()
    start=time.monotonic()
    loaded=set()
    print(f"Loading PyTorch weights on {torch.cuda.get_device_name(0)}",flush=True)
    # Materialize each weight directly on CUDA; no full CPU model or random model.
    for filename in dict.fromkeys(index["weight_map"].values()):
        with safe_open(str(destination/filename),framework="pt",device="cpu") as handle:
            weights={}
            for key in handle.keys():
                tensor=handle.get_tensor(key)
                weights[key]=tensor.to(device="cuda",dtype=expected[key].dtype)
                loaded.add(key)
            model.load_state_dict(weights,strict=False,assign=True)
            del weights
        gc.collect()
    if loaded!=set(expected):
        raise RuntimeError(f"Missing converted weights: {set(expected)-loaded}")
    # Nonpersistent buffers have no checkpoint entries. Recreate their exact
    # constructor values rather than copying uninitialized meta-device storage.
    for module in model.modules():
        if type(module).__name__=="GemmaRotaryEmbedding":
            module.__init__(config=module.config,device="cuda")
        for name,buffer in list(module._buffers.items()):
            if buffer is not None and buffer.device.type=="meta":
                if name=="position_ids":
                    module._buffers[name]=torch.arange(buffer.shape[-1],device="cuda",dtype=buffer.dtype).expand(buffer.shape)
                else:
                    raise RuntimeError(f"Unknown meta buffer {type(module).__name__}.{name}")
    if any(p.device.type!="cuda" for p in model.parameters()):
        raise RuntimeError("Some policy parameters are not on CUDA")
    model.eval()
    torch.cuda.synchronize()
    print(f"CUDA model ready in {time.monotonic()-start:.1f}s; allocated {torch.cuda.memory_allocated()/1024**3:.2f} GiB",flush=True)
    return model
