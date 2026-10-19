"""Verify recorded unassisted runs and assemble an evaluator-only comparison."""
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"runs/seed_diagnosis"


def summarize(directory):
    result=json.loads((directory/"results.json").read_text())
    diag=json.loads((directory/"diagnostics.json").read_text())
    records=[json.loads(line) for line in (directory/"telemetry.jsonl").read_text().splitlines()]
    exact=True
    vacuum=True
    for row in records:
        raw=np.load(directory/f"actions_{row['chunk']:03}.npy")[row["action_in_chunk"]]
        exact=exact and np.array_equal(raw,np.asarray(row["command"]["raw_action"]))
        vacuum=vacuum and (bool(raw[6]>0)==row["suction"])
    reader=imageio.get_reader(directory/"policy_rollout.mp4")
    try:
        metadata=reader.get_meta_data()
        frames=reader.count_frames()
    finally:
        reader.close()
    checks={"all_raw_policy_actions_passed_unchanged":bool(exact),
            "vacuum_matches_raw_policy_sign":bool(vacuum),
            "no_auxiliary_controller":result["controller_source"]=="frozen_pi05" and not result["controller_uses_card_ground_truth"] and not any("rgb_servo" in row or "lift_controller" in row for row in records),
            "gpu_policy":result["model_parameter_devices"]==["cuda:0"],
            "graphics_consistent":result["render_backend"]["renderer"]=="Intel(R) Arc(TM) Pro Graphics",
            "complete_twenty_seconds":len(records)==400 and len(result["chunks"])==80 and diag["physics_steps"]==10000 and frames==400 and metadata["fps"]==20}
    return {"directory":directory.relative_to(ROOT).as_posix(),"scene_seed":result["seed"],
            "noise_seed":result["policy_noise_seed"],"profile":result["simulation_profile"],
            "checks":checks,"closest_mm":diag["minimum_tcp_to_card_surface_m"]*1000,
            "contact_s":diag["cup_card_contact_steps"]*.002,
            "attachment_events":sum(row["event"]=="attached" for row in result["suction_events"]),
            "attached_s":diag["attached_steps"]*.002,
            "maximum_clearance_mm":diag["maximum_card_clearance_m"]*1000,
            "valid_hold_s":diag["maximum_continuous_valid_hold_s"],"success":result["task_success"]}


def video():
    sources=[OUT/"consistent_renderer/seed_0/baseline",OUT/"consistent_renderer/seed_1/baseline",
             OUT/"consistent_renderer/seed_42/baseline",OUT/"adapter_audit/suction_upright/scene_42_noise_0"]
    titles=["Scene 0 / noise 0", "Scene 1 / noise 1", "Scene 42 / noise 42", "Scene 42 / noise 0"]
    readers=[imageio.get_reader(path/"policy_rollout.mp4") for path in sources]
    summaries=[summarize(path) for path in sources]
    telemetry=[[json.loads(line) for line in (path/"telemetry.jsonl").read_text().splitlines()] for path in sources]
    font=ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf",19)
    small=ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf",16)
    target=OUT/"controlled_seed_comparison_20s.mp4"
    try:
        with imageio.get_writer(target,fps=20,macro_block_size=1) as writer:
            for index in range(400):
                frame=Image.new("RGB",(1280,1104),"#111821")
                draw=ImageDraw.Draw(frame)
                draw.text((16,7),f"Frozen pi0.5 on RTX 4070 | same scene/control settings | t={(index+1)/20:05.2f}s",font=font,fill="white")
                for panel,(reader,title,summary,records) in enumerate(zip(readers,titles,summaries,telemetry)):
                    x=(panel%2)*640;y=40+(panel//2)*532
                    row=records[index]
                    draw.text((x+12,y),f"{title} | closest {summary['closest_mm']:.2f} mm",font=font,fill="white")
                    draw.text((x+12,y+25),f"vacuum={row['suction']}  attached={row['attached']}  clearance={row['card_clearance_m']*1000:.1f} mm",font=small,fill="#98d8ef")
                    frame.paste(Image.fromarray(reader.get_data(index)),(x,y+52))
                writer.append_data(np.asarray(frame))
                if index==99:
                    frame.save(OUT/"controlled_seed_comparison_still.png")
    finally:
        for reader in readers:
            reader.close()
    print(target,flush=True)


def main():
    if "--video" in sys.argv:
        video()
    paths=sorted((OUT/"consistent_renderer").glob("seed_*/baseline"))+sorted((OUT/"adapter_audit").glob("*/scene_*"))
    rows=[summarize(path) for path in paths if (path/"results.json").exists()]
    report={"episodes":rows,"all_checks_pass":all(all(row["checks"].values()) for row in rows)}
    (OUT/"controlled_run_verification.json").write_text(json.dumps(report,indent=2))
    for row in rows:
        print(json.dumps({key:value for key,value in row.items() if key not in ["directory","checks"]}),flush=True)
    if not report["all_checks_pass"]:
        raise SystemExit(1)


if __name__=="__main__":
    main()
