"""Sequential 20-second GPU episodes; retain one frozen policy in VRAM."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pi05_policy
import run_policy
from scene_scenarios import SCENARIOS
from render_runtime import graphics_before_cuda


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--scenarios",nargs="+",choices=SCENARIOS,default=["baseline"])
    parser.add_argument("--output-dir",type=Path,default=ROOT/"runs/pi05_seed_sweep")
    parser.add_argument("--seeds",nargs="+",type=int,default=[0,1])
    args=parser.parse_args()
    # Serial use only: two simultaneous full models do not fit the 8 GB GPU.
    rows=[]
    with graphics_before_cuda():
        policy=pi05_policy.load_policy(backend="cuda")
        for seed in args.seeds:
            for scenario in args.scenarios:
                out=args.output_dir/f"seed_{seed}"/scenario
                episode_args=["--backend","cuda","--seconds","20","--seed",str(seed),"--scenario",scenario,"--output-dir",str(out)]
                run_policy.main(episode_args,policy=policy)
                result=json.loads((out/"results.json").read_text())
                diagnostics=json.loads((out/"diagnostics.json").read_text())
                rows.append({"seed":seed,"scenario":scenario,"controller_source":result["controller_source"],"task_success":result["task_success"],"simulated_seconds":result["simulated_seconds"],"policy_calls":len(result["chunks"]),"contact_s":diagnostics["cup_card_contact_steps"]*.002,"attachment_events":sum(event["event"]=="attached" for event in result["suction_events"]),"minimum_tcp_to_card_surface_m":diagnostics["minimum_tcp_to_card_surface_m"],"maximum_clearance_m":diagnostics["maximum_card_clearance_m"],"valid_hold_s":diagnostics["maximum_continuous_valid_hold_s"],"wall_seconds":result["wall_seconds"],"render_backend":result["render_backend"],"directory":str(out)})
                args.output_dir.mkdir(parents=True,exist_ok=True)
                (args.output_dir/"summary.json").write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows,indent=2),flush=True)


if __name__=="__main__":
    main()
