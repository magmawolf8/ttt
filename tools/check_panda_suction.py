"""Explicitly scripted adapter/mechanism checks, never learned-policy results."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from panda_card_sim import PandaCardEnv
from panda_suction import PandaSuctionEnv, PandaSuctionDiagnostics
from run_panda_card_mlx import verify_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/panda_suction_mechanics"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    env, reference = PandaSuctionEnv(42), PandaCardEnv(42)
    checks = {}
    report = {"controller": "scripted mechanics validation using evaluator card pose",
              "learned_policy_result": False, "checks": checks}
    try:
        env.reset()
        reference.reset()
        checks.update(verify_scene(env))
        checks["same_initial_arm_joints"] = bool(np.allclose(
            env.sim.data.qpos[env.robots[0]._ref_joint_pos_indexes],
            reference.sim.data.qpos[reference.robots[0]._ref_joint_pos_indexes], atol=1e-8, rtol=0))
        checks["same_TCP_position"] = bool(np.allclose(env.suction_tool.position,
            reference.sim.data.site_xpos[reference.robots[0].eef_site_id], atol=1e-10, rtol=0))
        checks["same_TCP_rotation"] = bool(np.allclose(env.suction_tool.rotation,
            reference.sim.data.site_xmat[reference.robots[0].eef_site_id].reshape(3, 3), atol=1e-10, rtol=0))
        for value, wanted in [(1., True), (-1., False), (0., False), (.2, True)]:
            obs, *_ = env.step([0.] * 6 + [value])
            checks[f"command_{value}_vacuum_{wanted}"] = env.suction_tool.suction == wanted
        checks["no_contact_does_not_attach"] = not bool(env.sim.data.eq_active[env.suction_tool.eq])
        checks["state_shape_8"] = env.observation_for_policy(obs, "Pick up the face-down card.")["observation/state"].shape == (8,)
        controller = env.robots[0].controller
        controller.update(force=True)
        before = controller.ee_pos.copy()
        controller.set_goal(np.array([.1, 0, 0, 0, 0, 0]))
        checks["action_x_0.1_requests_5mm"] = bool(np.allclose(controller.goal_pos - before, [.005, 0, 0], atol=1e-7))

        def move(target, steps, vacuum):
            start = env.suction_tool.position.copy()
            for fraction in np.linspace(0, 1, steps):
                desired = start + fraction * (np.asarray(target) - start)
                delta = np.clip((desired - env.suction_tool.position) / .05, -1, 1)
                env.step([*delta, 0, 0, 0, 1 if vacuum else -1])

        def pickup():
            env.physics_observer = None
            env.reset()
            card = env.sim.data.body_xpos[env.card_id].copy()  # Scripted check ONLY.
            diagnostics = PandaSuctionDiagnostics(env)
            env.physics_observer = diagnostics.physics_step
            move([card[0], card[1], .06], 60, False)
            # Gentle contact pressure for the compliant OSC controller. A
            # geometric surface target alone can stop short of physical contact.
            move([card[0], card[1], -.001], 60, True)
            move([card[0], card[1], -.001], 40, True)
            sealed = bool(env.sim.data.eq_active[env.suction_tool.eq])
            move([card[0], card[1], .12], 80, True)
            move([card[0], card[1], .12], 60, True)
            return sealed, diagnostics.report()

        sealed, score = pickup()
        checks["aligned_contact_seals"] = sealed
        checks["scripted_lift_passes_2s_hold"] = score["success"]
        report["scripted_pickup_diagnostics"] = score
        env.step([0.] * 6 + [-1.])
        checks["open_command_releases"] = not bool(env.sim.data.eq_active[env.suction_tool.eq])
        pickup()
        env.sim.data.xfrc_applied[env.card_id, 2] = -5.
        env.step([0.] * 6 + [1.])
        checks["overload_releases"] = any(event["event"] == "failed_holding_limit" for event in env.suction_tool.events)
        env.sim.data.xfrc_applied[:] = 0
        env.physics_observer = None
        env.reset()
        card = env.sim.data.body_xpos[env.card_id].copy()
        rotation = env.sim.data.body_xmat[env.card_id].reshape(3, 3)
        edge = card + rotation @ np.array([.029, 0, 0])
        move([edge[0], edge[1], .06], 60, False)
        move([edge[0], edge[1], -.001], 60, True)
        move([edge[0], edge[1], -.001], 40, True)
        checks["edge_overhang_rejected"] = not any(event["event"] == "attached" for event in env.suction_tool.events)
        print(json.dumps({"checks": len(checks), "failed": [k for k, v in checks.items() if not v],
                          "scripted_hold_s": score["maximum_continuous_valid_hold_s"]}, indent=2))
    finally:
        env.close()
        reference.close()
        (args.output_dir / "results.json").write_text(json.dumps(report, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
