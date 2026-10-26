"""Pretrained pi0.5 on its native single Panda + parallel gripper, using MLX.

Environment and preprocessing follow the pinned OpenPI LIBERO evaluation client.
Only RGB, proprioception and task text reach the policy. No scripted task actions.
"""
import argparse
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import subprocess
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from libero_runtime import ROOT, LIBERO_REVISION, configure_paths, configure_renderer, policy_observation

MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
             "libero_10": 520, "libero_90": 400}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dashboard(obs, prompt, step, success, latency):
    frame = Image.new("RGB", (960, 640), "#111821")
    draw = ImageDraw.Draw(frame)
    font = ImageFont.load_default(size=19)
    small = ImageFont.load_default(size=16)
    draw.text((16, 12), "Pretrained pi0.5 / MLX | Franka Panda + parallel gripper", font=font, fill="white")
    # Break at words so long benchmark instructions stay on-screen.
    words = prompt.split()
    lines = [""]
    for word in words:
        if len(lines[-1]) + len(word) > 90:
            lines.append("")
        lines[-1] += (" " if lines[-1] else "") + word
    for index, line in enumerate(lines):
        draw.text((16, 44 + 21 * index), line, font=small, fill="#91d9ff")
    base = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    frame.paste(Image.fromarray(base).resize((512, 512)), (16, 96))
    frame.paste(Image.fromarray(wrist).resize((384, 384)), (560, 96))
    draw.text((560, 493), "Wrist camera", font=font, fill="white")
    draw.text((560, 525), f"Step {step} | simulated time {step / 20:.2f}s", font=small, fill="white")
    draw.text((560, 550), f"Inference {latency:.2f}s / chunk", font=small, fill="white")
    draw.text((560, 578), "SUCCESS" if success else "Running learned policy", font=font,
              fill="#7bffad" if success else "#91d9ff")
    draw.text((16, 617), "20 Hz simulation; physics pauses during inference. Native LIBERO success predicate.", font=small, fill="#a9b3c2")
    return np.asarray(frame)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=MAX_STEPS, default="libero_spatial")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--init-start", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--policy-noise-seed", type=int, default=42)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/pi05_libero_mlx")
    parser.add_argument("--viewer", action="store_true", help="Launch with .venv-mlx/bin/mjpython on macOS")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.trials < 1 or args.init_start < 0 or not 1 <= args.replan_steps <= 10:
        parser.error("Positive trials, nonnegative initial state, and 1–10 replan steps required")
    max_steps = MAX_STEPS[args.suite] if args.max_steps is None else args.max_steps
    if max_steps < 1:
        parser.error("--max-steps must be positive")
    paths = configure_paths()
    revision = subprocess.check_output(["git", "-C", str(ROOT / "vendor/libero"), "rev-parse", "HEAD"], text=True).strip()
    if revision != LIBERO_REVISION:
        raise RuntimeError(f"Expected pinned LIBERO {LIBERO_REVISION}, found {revision}")
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    import torch  # LIBERO's initial-state files use Torch serialization; inference is MLX.
    from pi05_mlx import MLXPolicy
    import mlx.core as mx

    render_backend = configure_renderer()
    suite = benchmark.get_benchmark_dict()[args.suite]()
    if not 0 <= args.task < suite.n_tasks:
        parser.error(f"Task must be between 0 and {suite.n_tasks - 1}")
    task = suite.get_task(args.task)
    bddl = paths["bddl_files"] / task.problem_folder / task.bddl_file
    init_file = paths["init_states"] / task.problem_folder / task.init_states_file
    # Only this trusted, pinned upstream asset is unpickled. Torch >=2.6 changed
    # the default to weights_only=True, which rejects LIBERO's NumPy states.
    states = torch.load(init_file, map_location="cpu", weights_only=False)
    if args.init_start + args.trials > len(states):
        parser.error(f"Requested states exceed the {len(states)} available initial states")
    out = args.output_dir or ROOT / "runs" / ("libero_mlx_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=False)
    source = out / "source"
    source.mkdir()
    files = ["run_libero_mlx.py", "libero_runtime.py", "pi05_mlx.py"]
    for name in files:
        shutil.copyfile(ROOT / name, source / name)
    report = {"status": "starting", "policy": "pretrained_pi05_libero", "backend": "mlx",
              "robot": "single Franka Panda", "end_effector": "Panda parallel-jaw gripper",
              "controller": "native LIBERO OSC_POSE", "controller_uses_object_ground_truth": False,
              "learned_policy_only": True, "training_performed": False,
              "suite": args.suite, "task_id": args.task, "task": task.language,
              "libero_revision": revision, "bddl_sha256": sha256(bddl), "init_states_sha256": sha256(init_file),
              "source_sha256": {name: sha256(source / name) for name in files},
              "versions": {name: version(name) for name in ["mlx", "mujoco", "robosuite", "torch", "numpy"]},
              "environment_seed": args.seed, "control_hz": 20, "settle_steps": 10,
              "replan_steps": args.replan_steps, "max_steps": max_steps,
              "camera_render_size": 256, "policy_image_size": 224, "image_transform": "rotate_180",
              "render_backend": render_backend,
              "simulation_pauses_during_inference": True, "full_benchmark_evaluation": False, "episodes": []}
    (out / "results.json").write_text(json.dumps(report, indent=2))
    env = viewer = None
    try:
        np.random.seed(args.seed)
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
        env.seed(args.seed)
        policy = MLXPolicy(args.checkpoint, seed=args.policy_noise_seed)
        report["model"] = policy.metadata
        report["checkpoint_manifest_sha256"] = sha256(args.checkpoint / "manifest.json")
        for trial in range(args.trials):
            episode = out / f"episode_{trial:03}"
            episode.mkdir()
            env.reset()
            obs = env.set_init_state(states[args.init_start + trial])
            # Same stabilization as upstream: no task motion, gripper open.
            for _ in range(10):
                obs, _, _, _ = env.step([0.] * 6 + [-1.])
            if env.check_success():
                raise RuntimeError("Initial state is already successful; cannot count this trial")
            if args.viewer:
                import mujoco.viewer
                viewer = mujoco.viewer.launch_passive(env.sim.model._model, env.sim.data._data)
                viewer.cam.lookat[:] = [0., 0., .85]
                viewer.cam.distance = 1.7
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -30
                viewer.sync()
            noise_seed = args.policy_noise_seed + trial
            policy.reset_seed(noise_seed)
            latencies, commands, chunks = [], [], 0
            success = False
            began = time.monotonic()
            with imageio.get_writer(episode / "rollout.mp4", fps=20, macro_block_size=1) as video:
                video.append_data(dashboard(obs, task.language, 0, False, 0.))
                while len(commands) < max_steps:
                    inputs = policy_observation(obs, task.language)
                    np.savez_compressed(episode / f"observation_{chunks:03}.npz", **inputs)
                    tick = time.monotonic()
                    actions = np.asarray(policy.infer(inputs)["actions"])
                    latency = time.monotonic() - tick
                    latencies.append(latency)
                    if actions.shape != (10, 7) or not np.isfinite(actions).all():
                        raise ValueError(f"Invalid model action chunk: {actions.shape}")
                    np.save(episode / f"actions_{chunks:03}.npy", actions)
                    np.save(episode / f"noise_{chunks:03}.npy", policy.last_noise)
                    for index, action in enumerate(actions[:min(args.replan_steps, max_steps - len(commands))]):
                        obs, reward, done, info = env.step(action.tolist())
                        success = bool(env.check_success())
                        commands.append({"step": len(commands), "chunk": chunks, "action_in_chunk": index,
                                         "action": action.tolist(), "eef_pos": obs["robot0_eef_pos"].tolist(),
                                         "gripper_qpos": obs["robot0_gripper_qpos"].tolist(),
                                         "success": success,
                                         "object_positions_evaluator_only": {key: value.tolist() for key, value in obs.items()
                                             if key.endswith("_pos") and not key.startswith("robot") and "_to_" not in key}})
                        frame = dashboard(obs, task.language, len(commands), success, latency)
                        video.append_data(frame)
                        if viewer is not None:
                            viewer.sync()
                        if success:
                            break
                    print(json.dumps({"trial": trial, "chunk": chunks, "steps": len(commands),
                                      "inference_s": round(latency, 3), "success": success}), flush=True)
                    chunks += 1
                    if success:
                        break
                Image.fromarray(frame).save(episode / "final.png")
            if viewer is not None:
                viewer.close()
                viewer = None
            (episode / "telemetry.jsonl").write_text("\n".join(json.dumps(row) for row in commands) + "\n")
            result = {"trial": trial, "initial_state_index": args.init_start + trial,
                      "policy_noise_seed": noise_seed, "success": success, "steps": len(commands),
                      "simulated_seconds": len(commands) / 20, "wall_seconds": time.monotonic() - began,
                      "inference_latencies_s": latencies, "median_inference_s": float(np.median(latencies)),
                      "video": str((episode / "rollout.mp4").resolve())}
            report["episodes"].append(result)
            (out / "results.json").write_text(json.dumps(report, indent=2))
        report["status"] = "completed"
        report["successes"] = sum(ep["success"] for ep in report["episodes"])
        report["trials"] = args.trials
        print(json.dumps({"successes": report["successes"], "trials": args.trials, "output": str(out)}, indent=2))
    except BaseException as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if viewer is not None:
            viewer.close()
        if env is not None:
            env.close()
        report["mlx_peak_memory_bytes"] = mx.get_peak_memory()
        (out / "results.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
