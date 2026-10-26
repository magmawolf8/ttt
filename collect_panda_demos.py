"""Record explicitly scripted Panda suction demonstrations for supervised training.

The expert may read simulator state. Saved model inputs contain only the same
RGB cameras, state and prompt as inference. Entire episodes belong to one split.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from panda_suction import PandaSuctionEnv, PandaSuctionDiagnostics
from run_panda_card import verify_scene

PROMPT = "Pick up the face-down card."


def collect(env, seed):
    env.physics_observer = None
    env.scene_seed = seed
    obs = env.reset()
    for _ in range(4):
        obs, *_ = env.step([0.] * 6 + [-1.])
    diagnostics = PandaSuctionDiagnostics(env)
    env.physics_observer = diagnostics.physics_step
    card = env.sim.data.body_xpos[env.card_id].copy()  # Expert ONLY.
    records = {k: [] for k in ("image", "wrist_image", "state", "actions", "time")}
    telemetry, phases = [], []
    down = np.diag([1., -1., -1.])

    def move(target, steps, vacuum, phase):
        nonlocal obs
        start = env.suction_tool.position.copy()
        for fraction in np.linspace(0, 1, steps):
            desired = start + fraction * (np.asarray(target) - start)
            translation = np.clip((desired - env.suction_tool.position) / .05, -1, 1)
            orientation = Rotation.from_matrix(down @ env.suction_tool.rotation.T).as_rotvec()
            action = np.r_[translation, np.clip(orientation / .5, -.2, .2), 1. if vacuum else -1.].astype(np.float32)
            inputs = env.observation_for_policy(obs, PROMPT)
            for dest, source in [("image", "observation/image"), ("wrist_image", "observation/wrist_image"), ("state", "observation/state")]:
                records[dest].append(inputs[source].copy())
            records["actions"].append(action)
            records["time"].append(float(env.sim.data.time))
            phases.append(phase)
            telemetry.append(diagnostics.snapshot(env))
            obs, *_ = env.step(action)

    # Small deterministic timing variation; geometry/physics remain unchanged.
    rng = np.random.default_rng(seed + 98765)
    approach = int(rng.integers(45, 61))
    descend = int(rng.integers(45, 61))
    lift = int(rng.integers(60, 81))
    move([card[0], card[1], .06], approach, False, "approach")
    move([card[0], card[1], -.001], descend, True, "descend")
    move([card[0], card[1], -.001], 25, True, "seal")
    move([card[0], card[1], .12], lift, True, "lift")
    move([card[0], card[1], .12], 50, True, "hold")
    arrays = {key: np.asarray(value) for key, value in records.items()}
    arrays["prompt"] = np.array(PROMPT)
    score = diagnostics.report()
    env.physics_observer = None
    if arrays["actions"].shape[1:] != (7,) or not np.isfinite(arrays["actions"]).all():
        raise ValueError("Invalid expert actions")
    return arrays, {"seed": seed, "source": "scripted expert using simulator state; training labels only",
                    "frames": len(phases), "phases": phases, "diagnostics": score,
                    "suction_events": env.suction_tool.events.copy()}, telemetry


RECOVERY_MODES = ("normal", "hover", "lateral", "tilted", "reattach")


def collect_recovery(env, seed, mode):
    """Perturb the starting state, then record only expert corrections.

    Setup motions (including an intentional release) are not training labels.
    Object pose / attachment feedback are used by this data-generating expert,
    never by the learned rollout. Physics and inference observations are unchanged.
    """
    env.physics_observer = None
    env.scene_seed = seed
    obs = env.reset()
    for _ in range(4):
        obs, *_ = env.step([0.] * 6 + [-1.])
    rng = np.random.default_rng(seed + 98765)
    tool = env.suction_tool
    down = np.diag([1., -1., -1.])
    records = {k: [] for k in ("image", "wrist_image", "state", "actions", "time")}
    phases, telemetry = [], []
    diagnostics = None

    def card_position():
        return env.sim.data.body_xpos[env.card_id].copy()

    def command(desired, vacuum, phase=None, rotation=down):
        nonlocal obs
        translation = np.clip((np.asarray(desired) - tool.position) / .05, -1, 1)
        orientation = Rotation.from_matrix(rotation @ tool.rotation.T).as_rotvec()
        action = np.r_[translation, np.clip(orientation / .5, -.3, .3), 1. if vacuum else -1.].astype(np.float32)
        if phase is not None:
            inputs = env.observation_for_policy(obs, PROMPT)
            for dest, source in [("image", "observation/image"), ("wrist_image", "observation/wrist_image"), ("state", "observation/state")]:
                records[dest].append(inputs[source].copy())
            records["actions"].append(action)
            records["time"].append(float(env.sim.data.time))
            phases.append(phase)
            telemetry.append(diagnostics.snapshot(env))
        obs, *_ = env.step(action)

    def move(target, steps, vacuum=False, phase=None, rotation=down):
        start = tool.position.copy()
        for fraction in np.linspace(0, 1, steps):
            command(start + fraction * (np.asarray(target) - start), vacuum, phase, rotation)

    # Physical, unrecorded setup creates recoverable states; no teleportation.
    vacuum = False
    setup_attached = False
    if mode == "normal":
        # Half retain the exact evaluation start, half vary the elevated start.
        if seed % 2:
            move([rng.uniform(.40, .53), rng.uniform(-.11, .11), rng.uniform(.16, .27)], 35)
    else:
        card = card_position()
        offset = rng.uniform(-.012, .012, 2)
        height = rng.uniform(.02, .065)
        rotation = down
        if mode == "lateral":
            angle = rng.uniform(-np.pi, np.pi)
            offset = rng.uniform(.03, .065) * np.array([np.cos(angle), np.sin(angle)])
            height = rng.uniform(.018, .05)
        elif mode == "tilted":
            rotation = Rotation.from_rotvec(np.r_[rng.uniform(-.12, .12, 2), rng.uniform(-.3, .3)]).as_matrix() @ down
            height = rng.uniform(.022, .055)
        elif mode == "reattach":
            move([*card[:2], .04], 45)
            move([*card[:2], -.001], 55, True)
            for _ in range(25):
                if tool.data.eq_active[tool.eq]:
                    break
                command([*card_position()[:2], -.001], True)
            setup_attached = bool(tool.data.eq_active[tool.eq])
            if setup_attached:
                move([*card_position()[:2], .012], 25, True)
            command(tool.position.copy(), False)  # Actual physical release.
            move([*card_position()[:2], .04], 30, False)
            card = card_position()
            height = rng.uniform(.02, .055)
        vacuum = bool(rng.integers(0, 2))
        move([*(card[:2] + offset), height], 45, vacuum, rotation=rotation)

    diagnostics = PandaSuctionDiagnostics(env)
    env.physics_observer = diagnostics.physics_step
    initial = diagnostics.snapshot(env)
    contact_offset = rng.uniform(-.006, .006, 2)
    # Constant contact-directed behavior removes long, ambiguous pre-lift waits.
    for attempt in range(2):
        card = card_position()
        xy = card[:2] + contact_offset
        if tool.position[2] > .08:
            move([*xy, rng.uniform(.045, .065)], int(rng.integers(40, 66)), False, "approach")
        elif np.linalg.norm(tool.position[:2] - xy) > .018:
            move([*xy, max(.025, tool.position[2])], int(rng.integers(25, 41)), vacuum, "recover_align")
        # A servo target a few millimeters below the current cup produces a
        # consistent downward label until contact, including the failed hover zone.
        descent_step = rng.uniform(.008, .012)
        for _ in range(140):
            if tool.data.eq_active[tool.eq]:
                break
            xy = card_position()[:2] + contact_offset
            z = max(-.004, tool.position[2] - descent_step)
            command([*xy, z], True, "contact" if tool.position[2] < .009 else "descend")
        if tool.data.eq_active[tool.eq]:
            break
        move([*card_position()[:2], .03], 25, False, "recover_retry")
        contact_offset[:] = 0
    if tool.data.eq_active[tool.eq]:
        position = tool.position.copy()
        target_height = rng.uniform(.10, .16)
        move([*position[:2], target_height], int(rng.integers(65, 91)), True, "lift")
        move([*position[:2], target_height], 48, True, "hold")
    arrays = {key: np.asarray(value) for key, value in records.items()}
    arrays["prompt"] = np.array(PROMPT)
    score = diagnostics.report()
    env.physics_observer = None
    return arrays, {"seed":seed, "mode":mode, "setup_attachment_before_release":setup_attached,
                    "source":"scripted contact/recovery expert; setup perturbations excluded from labels",
                    "initial":initial, "frames":len(phases), "phases":phases,
                    "diagnostics":score, "suction_events":tool.events.copy()}, telemetry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-episodes", type=int, default=16)
    parser.add_argument("--validation-episodes", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--profile", choices=["pilot", "contact-recovery"], default="pilot")
    args = parser.parse_args()
    if min(args.train_episodes, args.validation_episodes) < 1:
        parser.error("Both episode counts must be positive")
    if args.train_episodes > 10000:
        parser.error("Training seed range would overlap validation")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"format": "panda-suction-demos-v1", "control_hz":20,
                "policy_inputs": ["image", "wrist_image", "state", "prompt"],
                "actions": "native OSC_POSE seven-value deltas; positive vacuum on",
                "source": "scripted expert; no scripted control used by evaluated policies",
                "episode_split": True, "episodes": [], "status": "collecting",
                "profile": args.profile,
                "source_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
                                  ["collect_panda_demos.py", "panda_suction.py", "panda_card_sim.py", "card_sim.py"]}}
    env = PandaSuctionEnv(args.seed_start)
    began = time.monotonic()
    try:
        manifest["scene_checks"] = verify_scene(env)
        for split, count, start in [("train", args.train_episodes, args.seed_start),
                                    ("validation", args.validation_episodes, args.seed_start + 10000)]:
            for index in range(count):
                if args.profile == "contact-recovery":
                    arrays, metadata, telemetry = collect_recovery(env, start + index, RECOVERY_MODES[index % len(RECOVERY_MODES)])
                else:
                    arrays, metadata, telemetry = collect(env, start + index)
                name = f"{split}_{index:04}"
                np.savez_compressed(args.output_dir / (name + ".npz"), **arrays)
                (args.output_dir / (name + "_evaluator.json")).write_text(json.dumps(telemetry))
                (args.output_dir / (name + ".json")).write_text(json.dumps(metadata, indent=2))
                entry = {"file": name + ".npz", "metadata": name + ".json", "split": split,
                         "seed": metadata["seed"], "frames": metadata["frames"],
                         "success": metadata["diagnostics"]["success"],
                         "mode": metadata.get("mode", "pilot"),
                         "sha256": hashlib.sha256((args.output_dir / (name + ".npz")).read_bytes()).hexdigest()}
                manifest["episodes"].append(entry)
                print(json.dumps(entry), flush=True)
                (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        manifest["status"] = "completed"
        manifest["wall_seconds"] = time.monotonic() - began
    finally:
        env.close()
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if not all(ep["success"] for ep in manifest["episodes"]):
        raise SystemExit("Some demonstrations failed; they are preserved and must be excluded from successful-action cloning")


if __name__ == "__main__":
    main()
