"""Audit a failed learned rollout against its successful demonstration.

Replay is explicitly scripted from recorded expert labels. Teacher-forced model
predictions use saved policy observations only. Neither is a pickup success by
the learned policy, and neither changes the deployed controller or suction rules.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--demonstration", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    diagnostics = json.loads((args.rollout / "diagnostics.json").read_text())
    telemetry = [json.loads(line) for line in (args.rollout / "telemetry.jsonl").read_text().splitlines()]
    metadata = json.loads(args.demonstration.with_suffix(".json").read_text())
    with np.load(args.demonstration) as data:
        demo = {key:data[key] for key in data.files}
    assert diagnostics["scene_seed"] == metadata["seed"]
    action_cache = {i:np.load(args.rollout / f"actions_{i:03}.npy") for i in {row["chunk"] for row in telemetry}}
    direct = all(np.array_equal(np.array(row["action"]), action_cache[row["chunk"]][row["action_in_chunk"]]) for row in telemetry)
    vacuum = all(row["suction"] == (row["action"][6] > 0) for row in telemetry)
    report = {"rollout":str(args.rollout), "demonstration":str(args.demonstration),
              "checkpoint":str(args.checkpoint), "scene_seed":metadata["seed"],
              "recorded_policy_actions_executed_exactly":direct,
              "vacuum_commands_match_action_head":vacuum,
              "learned_rollout_diagnostics":diagnostics}
    (args.output_dir / "audit.json").write_text(json.dumps(report, indent=2))

    from panda_suction import PandaSuctionEnv, PandaSuctionDiagnostics
    env = PandaSuctionEnv(metadata["seed"])
    try:
        obs = env.reset()
        for _ in range(4):
            obs, *_ = env.step([0.] * 6 + [-1.])
        score = PandaSuctionDiagnostics(env)
        env.physics_observer = score.physics_step
        matches = {"image":0, "wrist_image":0, "state":0}
        maximum_state_error = 0.
        for index, action in enumerate(demo["actions"]):
            actual = env.observation_for_policy(obs, str(demo["prompt"]))
            for key, source in [("image","observation/image"),("wrist_image","observation/wrist_image"),("state","observation/state")]:
                matches[key] += int(np.array_equal(actual[source], demo[key][index]))
            maximum_state_error = max(maximum_state_error, float(np.max(np.abs(actual["observation/state"]-demo["state"][index]))))
            obs, *_ = env.step(action)
        report["scripted_demonstration_replay"] = {"frames":len(demo["actions"]), "exact_observation_matches":matches,
                                                   "maximum_state_error":maximum_state_error, **score.report()}
        print(json.dumps({"replay_success":score.report()["success"], "matches":matches}), flush=True)
    finally:
        env.close()
        (args.output_dir / "audit.json").write_text(json.dumps(report, indent=2))

    from pi05_mlx import MLXPolicy, mx
    policy = MLXPolicy(args.checkpoint)
    records = []
    for index in range(0, len(demo["actions"]), 5):
        inputs = {"observation/image":demo["image"][index], "observation/wrist_image":demo["wrist_image"][index],
                  "observation/state":demo["state"][index], "prompt":str(demo["prompt"])}
        predicted = policy.infer(inputs)["actions"]
        count = min(5,len(demo["actions"])-index)
        target = demo["actions"][index:index+count]
        records.append({"frame":index, "phase":metadata["phases"][index],
                        "eef_m":demo["state"][index,:3].tolist(),
                        "predicted":predicted[:count].tolist(), "target":target.tolist()})
    (args.output_dir / "teacher_forced_predictions.json").write_text(json.dumps(records, indent=2))
    report["teacher_forced_by_phase"] = {}
    for phase in ["approach","descend","seal","lift","hold"]:
        selected = [row for row in records if row["phase"]==phase]
        predicted = np.concatenate([row["predicted"] for row in selected])
        target = np.concatenate([row["target"] for row in selected])
        report["teacher_forced_by_phase"][phase] = {"actions":len(predicted),
            "xyz_mae":np.mean(np.abs(predicted[:,:3]-target[:,:3]),axis=0).tolist(),
            "target_mean_dz":float(target[:,2].mean()), "predicted_mean_dz":float(predicted[:,2].mean()),
            "predicted_up_fraction":float(np.mean(predicted[:,2]>0)),
            "vacuum_sign_agreement":float(np.mean((predicted[:,6]>0)==(target[:,6]>0)))}

    # Check that batching two prefixes during training does not cross-wire images.
    caches, lengths = [], []
    for index in (0, min(100,len(demo["actions"])-1)):
        inputs = {"observation/image":demo["image"][index], "observation/wrist_image":demo["wrist_image"][index],
                  "observation/state":demo["state"][index], "prompt":str(demo["prompt"])}
        images,tokens=policy.prepare(inputs)
        cache,length=policy.model.encode_prefix(images,tokens)
        caches.append(cache); lengths.append(length)
    assert lengths[0]==lengths[1]
    noisy=mx.array(np.random.default_rng(1).normal(size=(2,10,32)).astype(np.float32))
    times=mx.array([.25,.75])
    stacked=[(mx.concatenate([caches[0][i][0],caches[1][i][0]],axis=0),
              mx.concatenate([caches[0][i][1],caches[1][i][1]],axis=0)) for i in range(18)]
    batched=policy.model.denoise(noisy,times,stacked,lengths[0])
    separate=mx.concatenate([policy.model.denoise(noisy[i:i+1],times[i:i+1],caches[i],lengths[i]) for i in range(2)],axis=0)
    difference=float(mx.max(mx.abs(batched-separate)).item())
    report["batched_vs_individual_velocity_max_abs"] = difference
    report["batching_check_passed"] = difference < 1e-4
    (args.output_dir / "audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["teacher_forced_by_phase"], indent=2), flush=True)
    print(json.dumps({"batching_max_abs_error":difference}), flush=True)


if __name__ == "__main__":
    main()
