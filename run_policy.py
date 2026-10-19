"""Closed-loop policy -> bounded controller -> MuJoCo, with reproducible diagnostics."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import traceback
import math
import shutil
import hashlib

import imageio.v2 as imageio
import numpy as np
import psutil

from card_sim import CardSim, ROOT
from episode_diagnostics import EpisodeDiagnostics, annotate_video_frame
from scene_scenarios import SCENARIOS, configure_scenario


def main(argv=None, policy=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--chunks",type=int,default=3)
    parser.add_argument("--seconds",type=float,default=None,help="Episode duration in simulated seconds, at 20 Hz")
    parser.add_argument("--image-transform",choices=["auto","native","mirror_x","rotate_180"],default="auto",help="Image convention; auto uses the simulation profile")
    parser.add_argument("--sim-profile",choices=["legacy","libero","suction_upright","suction_refined","suction_rgb256","reference_scale_audit"],default="suction_upright")
    parser.add_argument("--replan-steps",type=int,default=5,help="Maximum actions executed from each policy chunk")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--policy-noise-seed",type=int,default=None,help="Local CUDA sampling seed; defaults to the scene seed")
    parser.add_argument("--scenario",choices=SCENARIOS,default="baseline")
    parser.add_argument("--backend",choices=["cuda","cpu"],default="cuda")
    parser.add_argument("--output-dir",type=Path,default=None)
    parser.add_argument("--server",default=None,help="Optional OpenPI websocket server hostname")
    parser.add_argument("--port",type=int,default=8000)
    args=parser.parse_args(argv)
    if args.policy_noise_seed is not None and (args.server or args.backend!="cuda"):
        parser.error("--policy-noise-seed requires local CUDA inference")
    noise_seed=args.seed if args.policy_noise_seed is None else args.policy_noise_seed
    steps=round(args.seconds*20) if args.seconds is not None else args.chunks*5
    if steps<=0 or (args.seconds is not None and not math.isclose(steps/20,args.seconds)):
        parser.error("Duration must be a positive multiple of 0.05 simulated seconds")
    if not 1<=args.replan_steps<=10:
        parser.error("Replan steps must be between 1 and 10")
    out=args.output_dir or ROOT/"runs"/("pi05_remote" if args.server else "pi05_"+args.backend+"_"+args.sim_profile)
    out.mkdir(parents=True,exist_ok=True)
    sources=["run_policy.py","card_sim.py","scene_scenarios.py","episode_diagnostics.py",
             "pi05_policy.py","pi05_torch.py","tools/convert_pi05_torch.py","render_runtime.py",
             "assets/policy_profiles.json","assets/runtime_config.json","assets/scene.xml",
             "assets/card_back.png"]
    source_dir=out/"source"
    source_dir.mkdir(exist_ok=True)
    for name in sources:
        (source_dir/name).parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,source_dir/name)
    report={"status":"starting","policy":"pi05_libero","checkpoint":"gs://openpi-assets/checkpoints/pi05_libero","openpi_revision":subprocess.check_output(["git","-C",str(ROOT/"vendor/openpi"),"rev-parse","HEAD"],text=True).strip(),"seed":args.seed,"simulation_pauses_during_inference":True,"control_hz":20,"observations_contain_card_pose":False,"adaptation":"zero-shot experimental LIBERO -> UR7e and gripper -> suction transfer","initial_memory":psutil.virtual_memory()._asdict(),"chunks":[]}
    report["backend"]="remote" if args.server else args.backend
    report["source_sha256"]={name:hashlib.sha256((source_dir/name).read_bytes()).hexdigest() for name in sources}
    report["image_transform"]=args.image_transform
    report["policy_noise_seed"]=noise_seed
    sim=None
    video=None
    dashboard=None
    diagnostics=None
    try:
        sim=CardSim(args.seed,profile=args.sim_profile)
        report["scenario"]=configure_scenario(sim,args.scenario) if args.scenario!="baseline" else {"name":"baseline","initialization_only":True,"modifications":{}}
        args.image_transform=sim.policy_profile["image_transform"] if args.image_transform=="auto" else args.image_transform
        report.update(simulation_profile=args.sim_profile,policy_profile=sim.policy_profile,image_transform=args.image_transform)
        report["controller_source"]="frozen_pi05"
        report["controller_uses_card_ground_truth"]=False
        report["suction_config"]=dict(sim.suction_config)
        diagnostics=EpisodeDiagnostics(sim,out)
        sim.physics_observer=diagnostics.physics_step
        video=imageio.get_writer(out/"policy_rollout.mp4",fps=20,macro_block_size=1)
        dashboard=imageio.get_writer(out/"episode_dashboard.mp4",fps=20,macro_block_size=1)
        obs=sim.observation(image_transform=args.image_transform)
        from render_runtime import current_render_backend
        report["render_backend"]=current_render_backend()
        np.savez_compressed(out/"initial_observation.npz",**obs)
        if args.server:
            from openpi_client.websocket_client_policy import WebsocketClientPolicy
            policy=WebsocketClientPolicy(args.server,args.port)
        else:
            from pi05_policy import load_policy
            if policy is None:
                policy=load_policy(backend=args.backend)
            if args.backend=="cuda":
                import torch
                torch.manual_seed(noise_seed)
                report["gpu"]=torch.cuda.get_device_name(0)
                report["model_parameter_devices"]=sorted({str(p.device) for p in policy._model.parameters()})
                torch.cuda.reset_peak_memory_stats()
        print("Policy loaded; starting closed-loop inference",flush=True)
        initial_qpos=sim.data.qpos[:6].copy()
        start=time.monotonic()
        chunk_id=0
        while sim.command_count<steps:
            obs=sim.observation(image_transform=args.image_transform)
            diagnostics.record_visibility(sim,chunk_id)
            np.savez_compressed(out/f"observation_{chunk_id:03}.npz",**obs)
            t=time.monotonic()
            result=policy.infer(obs)
            if args.backend=="cuda" and not args.server:
                torch.cuda.synchronize()
            latency=time.monotonic()-t
            actions=np.asarray(result["actions"])
            if actions.ndim!=2 or actions.shape[1]!=7 or not np.isfinite(actions).all():
                raise ValueError(f"Invalid policy action chunk {actions.shape}")
            np.save(out/f"actions_{chunk_id:03}.npy",actions)
            before=sim.data.qpos[:6].copy()
            # Replan after 5 commands (0.25 simulated seconds).
            replan=args.replan_steps
            if sim.position[2]<sim.policy_profile.get("near_table_height_m",-1):
                replan=min(replan,sim.policy_profile["near_table_replan_steps"])
            execute=min(replan,steps-sim.command_count,len(actions))
            for index,action in enumerate(actions[:execute]):
                sim.apply_policy_action(action)
                sample=diagnostics.record_command(sim,chunk_id,index)
                external=sim.render(width=640,height=480)
                video.append_data(external)
                frame_obs=sim.observation(image_transform=args.image_transform)
                base_view=frame_obs["observation/image"]
                wrist_view=frame_obs["observation/wrist_image"]
                dashboard.append_data(annotate_video_frame(external,base_view,wrist_view,sample,latency,steps/20,args.image_transform))
            chunk={"index":chunk_id,"latency_seconds":latency,"action_shape":list(actions.shape),"executed":execute,"joint_displacement_rad":float(np.linalg.norm(sim.data.qpos[:6]-before)),"suction":sim.suction,"tcp_m":sim.position.tolist()}
            report["chunks"].append(chunk)
            print(json.dumps(chunk),flush=True)
            chunk_id+=1
        movement=float(np.linalg.norm(sim.data.qpos[:6]-initial_qpos))
        scored=diagnostics.save(sim)
        report.update(status="policy_controlling_robot" if movement>1e-4 else "no_observed_movement",joint_displacement_rad=movement,wall_seconds=time.monotonic()-start,simulated_seconds=sim.command_count/20,interventions=sim.interventions,suction_events=sim.events,card_attached_at_end=bool(sim.data.eq_active[sim.eq]),card_height_at_end_m=float(sim.data.xpos[sim.card,2]),task_success=scored["success"],failure_category=scored["failure_category"],full_acceptance_evaluation=False)
    except Exception as error:
        report.update(status="blocked",error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        print(report["traceback"],flush=True)
    finally:
        if video:
            video.close()
        if dashboard:
            dashboard.close()
        if sim:
            if diagnostics:
                diagnostics.save(sim)
            sim.close()
        report["final_memory"]=psutil.virtual_memory()._asdict()
        if args.backend=="cuda" and not args.server:
            import torch
            if torch.cuda.is_available():
                report["cuda_peak_allocated_bytes"]=torch.cuda.max_memory_allocated()
                report["cuda_peak_reserved_bytes"]=torch.cuda.max_memory_reserved()
        (out/"results.json").write_text(json.dumps(report,indent=2))
        print(json.dumps({k:v for k,v in report.items() if k not in ["traceback","interventions","chunks","initial_memory","final_memory"]},indent=2),flush=True)
    if report["status"]=="blocked":
        raise SystemExit(1)


if __name__=="__main__":
    main()
