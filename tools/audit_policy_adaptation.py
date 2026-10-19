"""Predeclared adapter/seed ablations with consistent graphics and frozen weights."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pi05_policy import load_policy
from render_runtime import graphics_before_cuda
import run_policy


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,default=ROOT/"runs/seed_diagnosis/adapter_audit_reproduction")
    args=parser.parse_args()
    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    plan=ROOT/"runs/seed_diagnosis/adapter_audit_plan.json"
    cases=json.loads(plan.read_text())["episodes"]
    shutil.copyfile(plan,out/"plan_used.json")
    shutil.copyfile(__file__,out/"audit_policy_adaptation_used.py")
    rows=[]
    # Must precede importing Torch as well as loading the model on this laptop.
    with graphics_before_cuda() as bootstrap:
        policy=load_policy(backend="cuda")
        import torch
        probes=[]
        for seed in [0,1,42]:
            saved=ROOT/f"runs/seed_diagnosis/consistent_renderer/seed_{seed}/baseline"
            with np.load(saved/"observation_000.npz") as source:
                obs={key:source[key] for key in source.files}
            obs["prompt"]=obs["prompt"].item()
            torch.manual_seed(seed)
            actions=np.asarray(policy.infer(obs)["actions"])
            probes.append({"seed":seed,"max_action_difference_from_saved":float(np.max(abs(actions-np.load(saved/"actions_000.npy"))))})
        (out/"initial_model_reuse_checks.json").write_text(json.dumps(probes,indent=2))
        for case in cases:
            seed=case["scene_seed"];noise=case["policy_noise_seed"];profile=case["profile"]
            directory=out/profile/f"scene_{seed}_noise_{noise}"
            print(f"AUDIT {profile}: scene={seed}, noise={noise}",flush=True)
            run_policy.main(["--backend","cuda","--seconds","20","--seed",str(seed),
                             "--policy-noise-seed",str(noise),"--sim-profile",profile,
                             "--output-dir",str(directory)],policy=policy)
            result=json.loads((directory/"results.json").read_text())
            diag=json.loads((directory/"diagnostics.json").read_text())
            rows.append({**case,"directory":directory.relative_to(ROOT).as_posix() if directory.is_relative_to(ROOT) else str(directory),
                         "controller_source":result["controller_source"],"render_backend":result["render_backend"],
                         "bootstrap_backend":bootstrap,"success":result["task_success"],
                         "closest_distance_mm":diag["minimum_tcp_to_card_surface_m"]*1000,
                         "contact_s":diag["cup_card_contact_steps"]*.002,
                         "eligible_s":diag["attachment_eligible_steps"]*.002,
                         "valid_hold_s":diag["maximum_continuous_valid_hold_s"],
                         "interventions":diag["controller_interventions"],
                         "simulated_seconds":result["simulated_seconds"],"policy_calls":len(result["chunks"])})
            (out/"results.json").write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows,indent=2),flush=True)


if __name__=="__main__":
    main()
