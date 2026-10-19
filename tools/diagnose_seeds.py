"""Frozen-policy seed diagnosis: independent scene/noise experiments, no guidance.

Object state is read only from completed evaluator records for analysis. Neither
counterfactual guidance nor a scripted action controller is used in episodes.
"""
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

OUT=ROOT/"runs/seed_diagnosis/consistent_crossed_reproduction"
SAVED={seed:ROOT/f"runs/seed_diagnosis/consistent_renderer/seed_{seed}/baseline"
       for seed in [0,1,42]}


def observation(path):
    with np.load(path) as saved:
        obs={key:saved[key] for key in saved.files}
    obs["prompt"]=obs["prompt"].item()
    return obs


def outcome(path):
    result=json.loads((path/"results.json").read_text())
    diag=json.loads((path/"diagnostics.json").read_text())
    return {"scene_seed":result["seed"],"policy_noise_seed":result["policy_noise_seed"],
            "directory":path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),"success":result["task_success"],
            "closest_distance_mm":diag["minimum_tcp_to_card_surface_m"]*1000,
            "contact_s":diag["cup_card_contact_steps"]*.002,
            "eligible_s":diag["attachment_eligible_steps"]*.002,
            "valid_hold_s":diag["maximum_continuous_valid_hold_s"],
            "minimum_tcp_height_mm":diag["minimum_tcp_height_m"]*1000}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,default=OUT)
    args=parser.parse_args()
    # The first GL context must precede even importing Torch on this laptop.
    with graphics_before_cuda():
        policy=load_policy(backend="cuda")
        import torch
        diagnose(policy,torch,args.output_dir.resolve())


def diagnose(policy,torch,output_dir):
    # Keep archived pre-fix records separate from new graphics-consistent probes.
    OUT=output_dir
    OUT.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(__file__,OUT/"diagnose_seeds_used.py")
    probes=[]
    for scene,path in SAVED.items():
        obs=observation(path/"observation_000.npz")
        first=json.loads((path/"telemetry.jsonl").read_text().splitlines()[0])
        target=np.asarray(first["card_m"][:2])-first["command"]["tcp_before_m"][:2]
        for noise_seed in [0,1,42]:
            torch.manual_seed(noise_seed)
            actions=np.asarray(policy.infer(obs)["actions"])
            mean=actions[:5,:3].mean(axis=0)
            cosine=float(mean[:2]@target/max(np.linalg.norm(mean[:2])*np.linalg.norm(target),1e-12))
            probes.append({"scene_seed":scene,"policy_noise_seed":noise_seed,
                           "mean_first_five_xyz_actions":mean.tolist(),
                           "initial_target_xy_error_evaluation_only_m":target.tolist(),
                           "initial_xy_direction_cosine_evaluation_only":cosine,
                           "saved_original_first_chunk_max_difference":float(np.max(abs(actions-np.load(path/"actions_000.npy")))) if scene==noise_seed else None})
    (OUT/"initial_input_probes.json").write_text(json.dumps(probes,indent=2))
    print("Initial fixed-input seed probes completed",flush=True)

    # Reconstruct the original Gaussian noise at the first differing observation.
    torch.manual_seed(42)
    for _ in range(22):
        noise=policy._model.sample_noise((1,10,32),"cuda")
    noise=noise.cpu().numpy()
    sensitivity=[]
    for name,path in [("original",ROOT/"runs/pi05_cuda_20s_upright"),("current",ROOT/"runs/pi05_seed42_regression_check")]:
        actions=np.asarray(policy.infer(observation(path/"observation_021.npz"),noise=noise)["actions"])
        np.save(OUT/f"pixel_probe_{name}_actions.npy",actions)
        sensitivity.append({"name":name,"max_difference_from_saved_actions":float(np.max(abs(actions-np.load(path/"actions_021.npy"))))})
    original=np.load(OUT/"pixel_probe_original_actions.npy")
    current=np.load(OUT/"pixel_probe_current_actions.npy")
    (OUT/"pixel_sensitivity.json").write_text(json.dumps({"chunk":21,"reconstructed_noise":True,"probes":sensitivity,"max_action_difference_between_inputs":float(np.max(abs(original-current)))},indent=2))

    cases=[(0,42),(1,42),(42,0),(42,1),(42,42)]
    rows=[outcome(path) for path in SAVED.values()]
    rows.append(outcome(ROOT/"runs/pi05_seed42_regression_check"))
    for scene,noise_seed in cases:
        out=OUT/"crossed"/f"scene_{scene}_noise_{noise_seed}"
        print(f"Starting scene={scene}, noise={noise_seed}",flush=True)
        run_policy.main(["--backend","cuda","--seconds","20","--seed",str(scene),
                         "--policy-noise-seed",str(noise_seed),"--output-dir",str(out)],policy=policy)
        rows.append(outcome(out))
        (OUT/"crossed_results.json").write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows,indent=2),flush=True)


if __name__=="__main__":
    main()
