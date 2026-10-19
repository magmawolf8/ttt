"""Summarize controlled frozen-policy experiments and make a 20 s comparison video."""
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
CASES=[("Original adapter","pi05_cuda_20s"),("Image convention only","pi05_cuda_20s_mirror"),("Calibrated cameras and controller","pi05_cuda_20s_calibrated"),("Calibrated and upright cup","pi05_cuda_20s_upright"),("RGB256 and slower near-table control","pi05_cuda_20s_refined"),("RGB256 and normal control schedule","pi05_cuda_20s_rgb256")]


def main():
    out=ROOT/"runs/calibration_audit"
    out.mkdir(parents=True,exist_ok=True)
    comparison=[]
    for label,name in CASES:
        p=ROOT/"runs"/name
        report=json.loads((p/"results.json").read_text())
        diagnostics=json.loads((p/"diagnostics.json").read_text())
        comparison.append({"condition":label,"directory":str(p),"seed":report["seed"],"simulated_seconds":report["simulated_seconds"],"commands":sum(c["executed"] for c in report["chunks"]),"policy_calls":len(report["chunks"]),"closest_tool_tip_to_card_surface_mm":1000*diagnostics["minimum_tcp_to_card_surface_m"],"contact_duration_s":.002*diagnostics["cup_card_contact_steps"],"eligible_duration_s":.002*diagnostics["attachment_eligible_steps"],"valid_hold_s":diagnostics["maximum_continuous_valid_hold_s"],"success":report["task_success"],"wall_seconds_excluding_loading":report["wall_seconds"],"median_inference_s":float(np.median([c["latency_seconds"] for c in report["chunks"]]))})
    (out/"episode_comparison.json").write_text(json.dumps(comparison,indent=2))
    original=ROOT/"runs/pi05_cuda_20s"
    corrected=ROOT/"runs/pi05_cuda_20s_upright"
    telemetry=[json.loads(x) for x in (corrected/"telemetry.jsonl").read_text().splitlines()]
    contacts=[row for row in telemetry if row["cup_card_contact"]]
    first=contacts[0]
    short_axis=abs(first["tcp_card_local_m"][0])
    normalizers=json.loads((ROOT/"models/pi05_libero/assets/physical-intelligence/libero/norm_stats.json").read_text())["norm_stats"]
    norm_audit={key:{"dimension":len(stats["mean"]),"finite":all(np.isfinite(np.array(v)).all() for v in stats.values()),"positive_quantile_spans":bool(np.all(np.array(stats["q99"])>np.array(stats["q01"])))} for key,stats in normalizers.items()}
    diagnosis={"weights_modified":False,"training_performed":False,"all_experiments_same_scene_seed":42,"not_a_success_rate_evaluation":True,"experiments":comparison,"verified_fixes":["Mirror native MuJoCo RGB horizontally to reproduce upstream LIBERO OpenGL image convention; pixel-exact audit passes for both cameras.","Use the benchmark external camera pose translated to this table and its 45 degree vertical FOV.","Correct the wrist camera's 180 degree tool-frame rotation and axial/radial mounting offset.","Make controller speed limits configurable and reduce excessive action truncation while retaining joint/workspace/actuator limits.","Provide an explicit 8 degree table-normal cup constraint without card-pose access.","Cache render contexts by resolution instead of recreating contexts for every video frame."],"interface_audit":{"normalizers":norm_audit,"proprioception_used_by_this_checkpoint":False,"zero_proprioception_max_output_change":0.0,"action_frames":"world-frame relative translation and left-multiplied relative rotation; all six direction tests pass","policy_schedule":"20 Hz commands, 10 predicted actions, five executed before replanning by default","tool0_frame":"UR manufacturer flange/tool0 transforms compose to identity relative to wrist_3; existing tool axis was correct","ground_truth_control_leak":False,"simulator_checks":"Three positive pickup/hold/release seeds and no-contact, edge, overload negative checks passed on the corrected profile."},"remaining_failure":{"stage":"valid interior suction attachment","contact_samples_at_command_rate":len(contacts),"interior_contact_samples_at_command_rate":sum(row["cup_inside_card_footprint"] for row in contacts),"first_contact_time_s":first["t"],"first_contact_short_axis_center_offset_mm":1000*short_axis,"maximum_valid_short_axis_center_offset_mm":22.5,"first_contact_cup_overhang_mm":1000*(short_axis-.0225),"first_contact_normal_error_degrees":first["normal_error_degrees"],"conclusion":"The corrected policy reaches and pushes the card, but its cup center overhangs the edge; the seal gate rejects attachment. The remaining observed error is spatial pickup precision for the suction tool. A gripper-to-vacuum transfer mismatch is a plausible contributor, not a fully isolated causal proof.","guardrails_preserved":"No distance snapping, relaxed edge seal, ground-truth pose guidance, or scripted policy lift was introduced."},"rejected_as_default":["Slower near-table motion with two-step replanning stalled above the card in this seed.","256-to-224 image resizing changed the trajectory but did not achieve attachment; retained as a separate reproducible profile."],"sources":{"openpi_preprocessing":"https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/libero/main.py","benchmark_controller":"https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/robosuite/controllers/config/osc_pose.json","benchmark_wrist_camera":"https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/robosuite/models/assets/robots/panda/robot.xml"}}
    (out/"diagnosis.json").write_text(json.dumps(diagnosis,indent=2))
    left=imageio.get_reader(original/"policy_rollout.mp4")
    right=imageio.get_reader(corrected/"policy_rollout.mp4")
    if left.count_frames()!=400 or right.count_frames()!=400:
        raise RuntimeError("Comparison inputs must each contain 400 frames")
    font=ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf",23)
    small=ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf",17)
    with imageio.get_writer(out/"before_after_20s.mp4",fps=20,macro_block_size=1) as writer:
        for index in range(400):
            canvas=Image.new("RGB",(1280,576),"#111821")
            canvas.paste(Image.fromarray(left.get_data(index)),(0,64))
            canvas.paste(Image.fromarray(right.get_data(index)),(640,64))
            draw=ImageDraw.Draw(canvas)
            draw.text((14,5),"Original adapter",font=font,fill="white")
            draw.text((654,5),"Corrected cameras/control + upright cup",font=font,fill="white")
            draw.text((14,37),"Closest approach: 233 mm | No card contact",font=small,fill="#d5dde6")
            draw.text((654,37),"Card contact | Edge overhang prevents attachment",font=small,fill="#91d9ff")
            draw.text((14,548),f"t = {(index+1)/20:05.2f} / 20.00 s   |   Same scene seed 42   |   GPU pi0.5   |   Simulated-time playback",font=small,fill="white")
            writer.append_data(np.asarray(canvas))
            if index in [65,399]:
                canvas.save(out/f"comparison_frame_{index:03d}.png")
    left.close(); right.close()
    print(json.dumps(comparison,indent=2))
    print(out/"before_after_20s.mp4")


if __name__=="__main__":
    main()
