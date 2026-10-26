"""Test unchanged pretrained pi05_libero weights on the Panda/card scene."""
import argparse
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import time

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from libero_runtime import ROOT, policy_observation
from panda_policy import load_panda_policy, peak_memory_bytes


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_scene(env):
    """Fail before inference if the replacement silently changes the object."""
    from panda_suction import PandaSuctionEnv
    original = mujoco.MjModel.from_xml_path(str(ROOT / "assets/scene.xml"))
    model = env.sim.model._model
    checks = {}
    for name in ["card_geom", "table", "floor"]:
        a, b = model.geom(name).id, original.geom(name).id
        for field in ["geom_type", "geom_size", "geom_pos", "geom_quat", "geom_friction",
                      "geom_solref", "geom_solimp", "geom_margin", "geom_condim",
                      "geom_contype", "geom_conaffinity"]:
            checks[f"{name}.{field}"] = bool(np.allclose(getattr(model, field)[a], getattr(original, field)[b], atol=1e-12, rtol=0))
    for field in ["body_mass", "body_inertia"]:
        checks[f"card.{field}"] = bool(np.allclose(getattr(model, field)[env.card_id],
                                     getattr(original, field)[original.body("card").id], atol=1e-12, rtol=0))
    checks["physics_timestep"] = model.opt.timestep == original.opt.timestep == .002
    checks["integrator"] = model.opt.integrator == original.opt.integrator
    if isinstance(env, PandaSuctionEnv):
        tool = env.suction_tool
        for name in ["cup_stem", "cup_contact"]:
            a = model.geom(env.robots[0].gripper.correct_naming(name)).id
            b = original.geom(name).id
            for field in ["geom_type", "geom_size", "geom_pos", "geom_friction", "geom_solref", "geom_solimp", "geom_margin"]:
                checks[f"original_{name}.{field}"] = bool(np.allclose(getattr(model, field)[a], getattr(original, field)[b], atol=1e-12, rtol=0))
        for field in ["body_mass", "body_inertia", "body_ipos"]:
            checks[f"original_cup.{field}"] = bool(np.allclose(getattr(model, field)[tool.cup],
                                                  getattr(original, field)[original.body("cup").id], atol=1e-12, rtol=0))
        checks["original_suction_constraint_inactive_at_reset"] = model.neq == 1 and not tool.data.eq_active[tool.eq]
        for field in ["eq_solref", "eq_solimp"]:
            checks[f"original_weld.{field}"] = bool(np.array_equal(getattr(model, field)[tool.eq],
                                                        getattr(original, field)[original.equality("suction_weld").id]))
        checks["seven_value_action_interface"] = env.action_dim == 7
    else:
        checks["no_attachment_constraints"] = model.neq == 0
    checks["single_seven_joint_Panda"] = len(env.robots) == 1 and len(env.robots[0].robot_model.joints) == 7
    if not all(checks.values()):
        raise RuntimeError(f"Scene differs unexpectedly: {[k for k, v in checks.items() if not v]}")
    return checks


def dashboard(obs, sample, success, seed, latency, tool="parallel", backend="mlx", finetuned=False):
    frame = Image.new("RGB", (960, 672), "#111821")
    draw = ImageDraw.Draw(frame)
    font = ImageFont.load_default(size=19)
    small = ImageFont.load_default(size=16)
    label = "Fine-tuned" if finetuned else "Pretrained"
    draw.text((16, 12), f"{label} pi0.5 / {backend.upper()} | Panda in the original card scene", font=font, fill="white")
    description = "Original suction cup | Close = vacuum ON; open = OFF" if tool == "suction" else "Physical parallel gripper | No attachment assist"
    draw.text((16, 45), "Pick up the face-down card. | " + description, font=small, fill="#91d9ff")
    for key, size, xy in [("agentview_image", 512, (16, 86)), ("robot0_eye_in_hand_image", 384, (560, 86))]:
        image = np.ascontiguousarray(obs[key][::-1, ::-1])
        frame.paste(Image.fromarray(image).resize((size, size)), xy)
    draw.text((560, 481), f"Seed {seed} | {sample['t']:.2f} simulated seconds", font=small, fill="white")
    draw.text((560, 507), f"Card lift above table: {1000 * sample['card_clearance_m']:.1f} mm", font=small, fill="white")
    gap = sample["tcp_to_card_surface_m"] if tool == "suction" else sample["finger_to_card_distance_m"]
    gap_label = "Cup-to-card gap" if tool == "suction" else "Finger-to-card distance"
    draw.text((560, 533), f"{gap_label}: {1000 * gap:.1f} mm", font=small, fill="#ffd18a")
    status = f"Vacuum: {'ON' if sample['suction'] else 'OFF'} | Attached: {sample['attached']}" if tool == "suction" else f"Both fingers contacting: {sample['bilateral_grasp']}"
    draw.text((560, 559), status, font=small, fill="white")
    draw.text((560, 585), f"Inference: {latency:.2f}s / chunk", font=small, fill="white")
    draw.text((560, 615), "SUCCESS" if success else "Pickup not yet achieved", font=font,
              fill="#7bffad" if success else "#91d9ff")
    draw.text((16, 647), "Success: physical grasp + lowest card point at least 50 mm above table for 2 seconds.", font=small, fill="#a9b3c2")
    return np.asarray(frame)


def main(argv=None, default_backend=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["cuda", "mlx"],
                        default=default_backend or ("mlx" if platform.system() == "Darwin" else "cuda"))
    parser.add_argument("--tool", choices=["suction", "parallel"], default="suction")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--policy-noise-seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, help="MLX checkpoint bundle, including its own normalizer and provenance")
    parser.add_argument("--seconds", type=float, default=20.)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--viewer", action="store_true", help="On macOS, launch using mjpython")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.checkpoint is not None and args.backend != "mlx":
        parser.error("--checkpoint currently requires --backend mlx")
    steps = round(args.seconds * 20)
    if steps <= 0 or not np.isclose(steps / 20, args.seconds) or not 1 <= args.replan_steps <= 10:
        parser.error("Use a positive multiple of 0.05 seconds and 1–10 replan steps")
    out = args.output_dir or ROOT / "runs" / ("panda_card_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=False)
    source = out / "source"
    files = ["run_panda_card.py", "run_panda_card_mlx.py", "panda_policy.py",
             "panda_card_sim.py", "panda_suction.py", "card_sim.py",
             "episode_diagnostics.py", "pi05_mlx.py", "pi05_policy.py", "pi05_torch.py",
             "tools/convert_pi05_torch.py", "libero_runtime.py", "render_runtime.py",
             "assets/scene.xml", "assets/card_back.png", "assets/policy_profiles.json", "assets/runtime_config.json"]
    for name in files:
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, source / name)
    report = {"status": "starting", "policy": "pretrained_pi05_libero", "training_performed": False,
              "learned_policy_only": True, "controller_uses_card_ground_truth": False,
              "robot": "single Franka Panda", "tool": args.tool, "backend": args.backend,
              "gripper": "original 18mm suction cup" if args.tool == "suction" else "native Panda parallel-jaw gripper",
              "controller": "native robosuite OSC_POSE", "scripted_control": False,
              "attachment_model": "original finite-strength conditional suction weld" if args.tool == "suction" else "physical finger contacts",
              "prompt": "Pick up the face-down card.", "scene_seeds": args.seeds,
              "policy_noise_seed": args.policy_noise_seed, "seconds_per_trial": args.seconds,
              "replan_steps": args.replan_steps, "physics_hz": 500, "control_hz": 20,
              "simulation_pauses_during_inference": True,
              "source_sha256": {name: sha256(source / name) for name in files},
              "versions": {name: version(name) for name in
                           ["mlx" if args.backend == "mlx" else "torch", "mujoco", "robosuite", "numpy"]},
              "episodes": []}
    (out / "results.json").write_text(json.dumps(report, indent=2))
    env = viewer = policy = None
    try:
        from panda_card_sim import PandaCardEnv, PandaCardDiagnostics
        from panda_suction import PandaSuctionEnv, PandaSuctionDiagnostics
        env_class = PandaSuctionEnv if args.tool == "suction" else PandaCardEnv
        diagnostics_class = PandaSuctionDiagnostics if args.tool == "suction" else PandaCardDiagnostics
        env = env_class(args.seeds[0])
        report["scene_preservation_checks"] = verify_scene(env)
        report["render_backend"] = env.render_backend
        # Construct/render the simulator before importing Torch: hybrid Windows
        # laptops otherwise select a different OpenGL adapter and camera pixels.
        env.sim.render(camera_name="agentview", width=224, height=224)
        from render_runtime import current_render_backend
        report["graphics"] = current_render_backend()
        policy = load_panda_policy(args.backend, seed=args.policy_noise_seed, checkpoint=args.checkpoint)
        report["model"] = policy.metadata
        finetuned = "finetuning" in policy.metadata
        report["training_performed"] = finetuned
        report["policy"] = "finetuned_pi05_panda_suction" if finetuned else "pretrained_pi05_libero"
        report["checkpoint_manifest_sha256"] = policy.metadata["checkpoint_manifest_sha256"]
        if args.tool == "suction":
            report["suction_config"] = env.suction_tool.suction_config
            report["action_mapping"] = {"shape": [10, 7], "first_six": "unchanged native OSC_POSE deltas",
                                        "gripper": "positive -> vacuum on, nonpositive -> vacuum off",
                                        "tcp": "cup face coincides with native Panda grip site",
                                        "state": "8 values; virtual fingers encode vacuum command, not seal feedback"}
        for trial, seed in enumerate(args.seeds):
            episode = out / f"episode_{trial:03}_seed_{seed}"
            episode.mkdir()
            env.physics_observer = None
            env.scene_seed = seed
            obs = env.reset()
            # Match the original scene's 100 physics steps of initial settling.
            for _ in range(4):
                obs, _, _, _ = env.step([0.] * 6 + [-1.])
            diagnostics = diagnostics_class(env)
            env.physics_observer = diagnostics.physics_step
            initial = diagnostics.snapshot(env)
            (episode / "initial_state_evaluator_only.json").write_text(json.dumps(initial, indent=2))
            (episode / "scene.xml").write_text(env.model.get_xml())
            if args.viewer:
                import mujoco.viewer
                viewer = mujoco.viewer.launch_passive(env.sim.model._model, env.sim.data._data)
                viewer.cam.lookat[:] = [.4, 0, .2]
                viewer.cam.distance = 1.3
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -30
                viewer.sync()
            policy.reset_seed(args.policy_noise_seed)
            commands, latencies = [], []
            chunk = 0
            began = time.monotonic()
            with imageio.get_writer(episode / "rollout.mp4", fps=20, macro_block_size=1) as video:
                first_frame = dashboard(obs, initial, False, seed, 0., args.tool, args.backend, finetuned)
                Image.fromarray(first_frame).save(episode / "initial.png")
                video.append_data(first_frame)
                while len(commands) < steps:
                    inputs = env.observation_for_policy(obs, report["prompt"]) if args.tool == "suction" else policy_observation(obs, report["prompt"])
                    np.savez_compressed(episode / f"observation_{chunk:03}.npz", **inputs)
                    tick = time.monotonic()
                    actions = np.asarray(policy.infer(inputs)["actions"])
                    latency = time.monotonic() - tick
                    latencies.append(latency)
                    if actions.shape != (10, 7) or not np.isfinite(actions).all():
                        raise ValueError("Invalid policy action chunk")
                    np.save(episode / f"actions_{chunk:03}.npy", actions)
                    np.save(episode / f"noise_{chunk:03}.npy", policy.last_noise)
                    for index, action in enumerate(actions[:min(args.replan_steps, steps - len(commands))]):
                        obs, _, _, _ = env.step(action)
                        sample = diagnostics.snapshot(env)
                        sample.update(step=len(commands), chunk=chunk, action_in_chunk=index, action=action.tolist())
                        commands.append(sample)
                        frame = dashboard(obs, sample, diagnostics.maximum_hold >= 2., seed, latency, args.tool, args.backend, finetuned)
                        video.append_data(frame)
                        if viewer is not None:
                            viewer.sync()
                    print(json.dumps({"seed": seed, "chunk": chunk, "step": len(commands),
                                      "clearance_mm": round(sample["card_clearance_m"] * 1000, 3),
                                      "contact_steps": diagnostics.contact_steps,
                                      "grasp_or_attachment_steps": diagnostics.attachment_steps if args.tool == "suction" else diagnostics.bilateral_steps,
                                      "hold_s": round(diagnostics.maximum_hold, 3)}), flush=True)
                    chunk += 1
                Image.fromarray(frame).save(episode / "final.png")
            if viewer is not None:
                viewer.close()
                viewer = None
            env.physics_observer = None
            (episode / "telemetry.jsonl").write_text("\n".join(json.dumps(row) for row in commands) + "\n")
            result = {"scene_seed": seed, "policy_noise_seed": args.policy_noise_seed,
                      "steps": len(commands), "wall_seconds": time.monotonic() - began,
                      "median_inference_s": float(np.median(latencies)),
                      "initial": initial, **diagnostics.report()}
            if args.tool == "suction":
                result["suction_events"] = env.suction_tool.events.copy()
            (episode / "diagnostics.json").write_text(json.dumps(result, indent=2))
            report["episodes"].append(result)
            (out / "results.json").write_text(json.dumps(report, indent=2))
        report.update(status="completed", successes=sum(e["success"] for e in report["episodes"]),
                      completed_trials=len(report["episodes"]))
        print(json.dumps({"successes": report["successes"], "trials": report["completed_trials"], "output": str(out)}))
    except BaseException as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if viewer is not None:
            viewer.close()
        if env is not None:
            env.close()
        if policy is not None:
            report[f"{args.backend}_peak_memory_bytes"] = peak_memory_bytes(args.backend)
        (out / "results.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
