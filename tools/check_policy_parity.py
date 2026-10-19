"""Inference-only original JAX CPU versus converted CUDA checks across seeds."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--prepare-only",action="store_true")
    args=parser.parse_args()
    out=ROOT/"runs/seed_diagnosis/policy_parity"
    out.mkdir(parents=True,exist_ok=True)
    if args.prepare_only:
        import torch
        for seed in [0,1,42]:
            torch.manual_seed(seed)
            noise=torch.normal(mean=0.,std=1.,size=(1,10,32),dtype=torch.float32,device="cuda")
            np.save(out/f"noise_{seed}.npy",noise.cpu().numpy())
        (out/"plan.json").write_text(json.dumps({"seeds":[0,1,42],"inference_only":True,"training":False,
            "expected_gpu_actions":"recorded first action chunks from consistent_renderer runs",
            "reference":"original pi05_libero checkpoint through upstream JAX implementation",
            "pass_limits":{"relative_l2":.02,"maximum_xyz_error_mm":1.,"executed_gripper_signs_equal":True}},indent=2))
        print("Prepared exact CUDA Gaussian samples; no model loaded",flush=True)
        return
    from pi05_policy import load_policy
    policy=load_policy(backend="cpu")
    rows=[]
    for seed in [0,1,42]:
        source=ROOT/f"runs/seed_diagnosis/consistent_renderer/seed_{seed}/baseline"
        with np.load(source/"observation_000.npz") as saved:
            obs={key:saved[key] for key in saved.files}
        obs["prompt"]=obs["prompt"].item()
        start=time.monotonic()
        actual=np.asarray(policy.infer(obs,noise=np.load(out/f"noise_{seed}.npy"))["actions"])
        expected=np.load(source/"actions_000.npy")
        np.save(out/f"jax_actions_{seed}.npy",actual)
        diff=actual-expected
        relative=float(np.linalg.norm(diff)/max(np.linalg.norm(expected),1e-12))
        xyz_mm=float(np.max(abs(diff[:,:3]))*.05*1000)
        signs=bool(np.array_equal(actual[:5,6]>0,expected[:5,6]>0))
        row={"seed":seed,"relative_l2":relative,"maximum_action_difference":float(np.max(abs(diff))),
             "maximum_xyz_error_mm":xyz_mm,"executed_gripper_signs_equal":signs,
             "wall_seconds":time.monotonic()-start,"pass":relative<.02 and xyz_mm<1. and signs}
        rows.append(row)
        (out/"results.json").write_text(json.dumps(rows,indent=2))
        print(json.dumps(row),flush=True)
    if not all(row["pass"] for row in rows):
        raise SystemExit(1)


if __name__=="__main__":
    main()
