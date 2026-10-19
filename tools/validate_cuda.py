"""Compare CUDA conversion with the saved JAX output using identical inputs/noise."""
import json
from pathlib import Path
import sys
import time

import jax
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pi05_policy import load_policy


def main():
    baseline=ROOT/"runs/pi05"
    recorded=np.load(baseline/"observation_000.npz")
    obs={key:recorded[key] for key in recorded.files}
    obs["prompt"]=obs["prompt"].item()
    # Original Policy starts at key(0), splits once, then samples this exact noise.
    _,sample_key=jax.random.split(jax.random.key(0))
    noise=np.asarray(jax.random.normal(sample_key,(1,10,32))).copy()
    reference=np.load(baseline/"actions_000.npy")
    policy=load_policy(backend="cuda")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    actual=policy.infer(obs,noise=noise)["actions"]
    torch.cuda.synchronize()
    error=actual-reference
    report={"same_observation":True,"same_flow_noise":True,"all_parameters_cuda":all(p.device.type=="cuda" for p in policy._model.parameters()),"max_absolute_action_difference":float(np.max(np.abs(error))),"mean_absolute_action_difference":float(np.mean(np.abs(error))),"relative_l2_action_difference":float(np.linalg.norm(error)/np.linalg.norm(reference)),"max_absolute_difference_per_dimension":np.max(np.abs(error),axis=0).tolist(),"warm_inference_seconds":[]}
    for _ in range(5):
        start=time.perf_counter()
        result=policy.infer(obs,noise=noise)["actions"]
        torch.cuda.synchronize()
        report["warm_inference_seconds"].append(time.perf_counter()-start)
        if not np.isfinite(result).all():
            raise RuntimeError("Nonfinite CUDA outputs")
    report["median_warm_inference_seconds"]=float(np.median(report["warm_inference_seconds"]))
    report["cuda_peak_allocated_bytes"]=torch.cuda.max_memory_allocated()
    report["cuda_peak_reserved_bytes"]=torch.cuda.max_memory_reserved()
    # Allow modest numerical differences between the two upstream bf16 backends.
    report["conversion_check_passed"]=report["relative_l2_action_difference"]<.05 and report["all_parameters_cuda"]
    (ROOT/"runs/pi05_cuda/conversion_validation.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    if not report["conversion_check_passed"]:
        raise SystemExit(1)


if __name__=="__main__":
    main()
