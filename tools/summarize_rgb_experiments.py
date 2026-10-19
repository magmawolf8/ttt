"""Collect development outcomes and make a clearly labelled comparison video."""
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]


def summarize(directory):
    result=json.loads((directory/"results.json").read_text())
    diagnostics=json.loads((directory/"diagnostics.json").read_text())
    telemetry=[json.loads(line) for line in (directory/"telemetry.jsonl").read_text().splitlines()]
    attached=[row for row in telemetry if row["attached"]]
    first=attached[0] if attached else None
    scenario=result.get("scenario",{}).get("name","baseline")
    negative=scenario in ("red_card","missing")
    setup_bug=directory==ROOT/"runs/rgb_scenarios/heavy_slippery"
    seal_drift=None if first is None else max(float(np.linalg.norm(np.asarray(row["tcp_m"][:2])-first["tcp_m"][:2])) for row in attached)
    estimate_errors=[]
    for row in telemetry:
        estimate=row.get("rgb_servo",{}).get("estimated_card_xy_m")
        if estimate is not None and row["t"]<3:
            estimate_errors.append(float(np.linalg.norm(np.asarray(estimate)-row["card_m"][:2])))
    return {"directory":directory.relative_to(ROOT).as_posix(),"controller_source":result.get("controller_source","frozen_pi05"),
            "scenario":scenario,"seed":result["seed"],
            "valid_controlled_scenario":not setup_bug,
            "setup_note":"Invalid mass ablation: mj_setConst reset live state; retained as a diagnosed setup bug" if setup_bug else None,
            "expected_behavior":"stationary_rejection" if negative else "pickup",
            "expected_behavior_passed":None if setup_bug else bool(not result["task_success"] and not result["suction_events"] and result["joint_displacement_rad"]<1e-4 and diagnostics["suction_on_fraction"]==0) if negative else result["task_success"],
            "success":result["task_success"],"simulated_seconds":result["simulated_seconds"],
            "commands":len(telemetry),"policy_calls":len(result["chunks"]),
            "wall_seconds":result["wall_seconds"],"median_inference_s":float(np.median([row["latency_seconds"] for row in result["chunks"]])),
            "contact_s":diagnostics["cup_card_contact_steps"]*.002,"eligible_s":diagnostics["attachment_eligible_steps"]*.002,
            "attached_s":diagnostics["attached_steps"]*.002,"maximum_clearance_mm":diagnostics["maximum_card_clearance_m"]*1000,
            "continuous_valid_hold_s":diagnostics["maximum_continuous_valid_hold_s"],
            "attachment_events":sum(event["event"]=="attached" for event in result["suction_events"]),
            "release_events":sum(event["event"]!="attached" for event in result["suction_events"]),
            "first_attached_command_s":None if first is None else first["t"],
            "first_attached_card_local_tcp_mm":None if first is None else (np.asarray(first["tcp_card_local_m"])*1000).tolist(),
            "maximum_attached_tcp_xy_excursion_mm":None if seal_drift is None else seal_drift*1000,
            "early_estimate_max_error_mm":None if not estimate_errors else max(estimate_errors)*1000,
            "cuda_peak_allocated_gib":result.get("cuda_peak_allocated_bytes",0)/1024**3}


def main():
    out=ROOT/"runs/rgb_experiments"
    out.mkdir(parents=True,exist_ok=True)
    paths=[ROOT/"runs"/name for name in ["pi05_cuda_20s_upright","pi05_rgb_centering_seed42","pi05_rgb_centering_latched_seed42","pi05_rgb_pickup_seed42"]]
    paths+=sorted((ROOT/"runs/rgb_scenarios").glob("*/results.json"))
    paths+=sorted((ROOT/"runs/rgb_followup").glob("*/results.json"))
    paths=[p.parent if p.name=="results.json" else p for p in paths]
    rows=[summarize(p) for p in paths if (p/"results.json").exists()]
    (out/"summary.json").write_text(json.dumps(rows,indent=2))
    # Same scene, original learned approach vs the explicit assisted completion.
    left=imageio.get_reader(ROOT/"runs/pi05_cuda_20s_upright/policy_rollout.mp4")
    right=imageio.get_reader(ROOT/"runs/pi05_rgb_pickup_seed42/policy_rollout.mp4")
    font=ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf",22)
    small=ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf",17)
    if left.count_frames()!=400 or right.count_frames()!=400:
        raise RuntimeError("Expected full 20-second comparison episodes")
    with imageio.get_writer(out/"edge_fix_comparison.mp4",fps=20,macro_block_size=1) as writer:
        for i in range(400):
            frame=Image.new("RGB",(1280,576),"#111821")
            frame.paste(Image.fromarray(left.get_data(i)),(0,64))
            frame.paste(Image.fromarray(right.get_data(i)),(640,64))
            draw=ImageDraw.Draw(frame)
            draw.text((12,5),"Frozen pi0.5 + upright adapter",font=font,fill="white")
            draw.text((652,5),"Hybrid: pi0.5 + RGB centering + seal/lift",font=font,fill="white")
            draw.text((12,36),"Edge overhang; no valid seal",font=small,fill="#d2d9e1")
            draw.text((652,36),"No card pose input | Weights and suction gates unchanged",font=small,fill="#91d9ff")
            draw.text((12,548),f"t={(i+1)/20:05.2f}/20.00 s | Seed 42 | GPU inference; playback follows simulated time",font=small,fill="white")
            writer.append_data(np.asarray(frame))
            if i==240:
                frame.save(out/"comparison_frame.png")
    left.close();right.close()
    print(json.dumps(rows,indent=2))


if __name__=="__main__":
    main()
