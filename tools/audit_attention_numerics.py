"""Fixed-input inference ablations; no weight reads, edits, or controller changes."""
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import torch
    from transformers.models.gemma import modeling_gemma
    from pi05_policy import load_policy

    out=ROOT/"runs/seed_diagnosis/attention_numerics"
    out.mkdir(parents=True,exist_ok=True)
    policy=load_policy(backend="cuda")
    original=modeling_gemma.eager_attention_forward
    original_tf32=torch.backends.cuda.matmul.allow_tf32

    def attention(prescale):
        def forward(module,query,key,value,attention_mask,scaling,dropout=0.,**kwargs):
            keys=modeling_gemma.repeat_kv(key,module.num_key_value_groups)
            values=modeling_gemma.repeat_kv(value,module.num_key_value_groups)
            # JAX scales the bf16 query before its fp32-logit dot product.
            q=(query*scaling).float() if prescale else query.float()
            logits=torch.matmul(q,keys.float().transpose(2,3))
            if not prescale:
                logits=logits*scaling
            if attention_mask is not None:
                logits=logits+attention_mask[:,:,:,:keys.shape[-2]]
            probs=torch.softmax(logits,dim=-1,dtype=torch.float32).to(query.dtype)
            probs=torch.nn.functional.dropout(probs,p=dropout,training=module.training)
            output=torch.matmul(probs,values).transpose(1,2).contiguous()
            return output,probs
        return forward

    rows=[]
    try:
        for name,implementation in [("upstream_default",original),
                                     ("float32_logits_postscale",attention(False)),
                                     ("float32_logits_prescale",attention(True))]:
            modeling_gemma.eager_attention_forward=implementation
            torch.backends.cuda.matmul.allow_tf32=original_tf32 if name=="upstream_default" else False
            for seed in [0,1,42]:
                saved=ROOT/f"runs/seed_diagnosis/consistent_renderer/seed_{seed}/baseline"
                with np.load(saved/"observation_000.npz") as source:
                    obs={key:source[key] for key in source.files}
                obs["prompt"]=obs["prompt"].item()
                noise=np.load(ROOT/f"runs/seed_diagnosis/policy_parity/noise_{seed}.npy")
                start=time.monotonic()
                actual=np.asarray(policy.infer(obs,noise=noise)["actions"])
                reference=np.load(ROOT/f"runs/seed_diagnosis/policy_parity/jax_actions_{seed}.npy")
                previous=np.load(saved/"actions_000.npy")
                np.save(out/f"{name}_seed_{seed}.npy",actual)
                error=actual-reference
                row={"condition":name,"seed":seed,
                     "relative_l2_vs_jax":float(np.linalg.norm(error)/np.linalg.norm(reference)),
                     "first_five_relative_l2_vs_jax":float(np.linalg.norm(error[:5])/np.linalg.norm(reference[:5])),
                     "maximum_xyz_difference_mm":float(np.max(abs(error[:,:3]))*50),
                     "first_five_maximum_xyz_difference_mm":float(np.max(abs(error[:5,:3]))*50),
                     "all_gripper_signs_equal":bool(np.array_equal(actual[:,6]>0,reference[:,6]>0)),
                     "executed_gripper_signs_equal":bool(np.array_equal(actual[:5,6]>0,reference[:5,6]>0)),
                     "maximum_difference_from_recorded_cuda":float(np.max(abs(actual-previous))),
                     "wall_seconds":time.monotonic()-start}
                rows.append(row)
                (out/"results.json").write_text(json.dumps(rows,indent=2))
                print(json.dumps(row),flush=True)
    finally:
        modeling_gemma.eager_attention_forward=original
        torch.backends.cuda.matmul.allow_tf32=original_tf32


if __name__=="__main__":
    main()
