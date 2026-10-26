"""Local MLX action-expert fine-tuning with OpenPI's flow-matching objective.

Custom MLX trainer, not the upstream NVIDIA trainer. Vision/language weights
remain frozen. Original normalization is retained deliberately for continued
training in the identical Panda OSC action space. No simulator state is loaded.
"""
import argparse
from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

from pi05_mlx import MLXPolicy, Pi05MLX, mx
import mlx.optimizers as optim
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def is_action_parameter(name):
    return name.startswith(("action_", "time_mlp_")) or (
        name.startswith("PaliGemma/llm/") and any(part.endswith("_1") for part in name.split("/")))


def action_chunk(actions, index, horizon=10):
    """Repeat the final holding action; never read across episode boundaries."""
    return actions[np.minimum(np.arange(index, index + horizon), len(actions) - 1)]


def load_dataset(path, stride):
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["status"] != "completed" or manifest["format"] != "panda-suction-demos-v1":
        raise ValueError("Require a completed, identified Panda demonstration dataset")
    episodes, splits = {}, {"train": [], "validation": []}
    seeds = {key: set() for key in splits}
    for entry in manifest["episodes"]:
        if not entry["success"]:
            continue  # Failure records remain on disk but are not cloned.
        file = path / entry["file"]
        if sha256(file) != entry["sha256"]:
            raise ValueError(f"Dataset hash mismatch: {file}")
        with np.load(file) as data:
            episode = {key: data[key] for key in ("image", "wrist_image", "state", "actions", "prompt")}
        episode["phases"] = json.loads((path / entry["metadata"]).read_text())["phases"]
        count = len(episode["actions"])
        if episode["actions"].shape != (count, 7) or not np.isfinite(episode["actions"]).all():
            raise ValueError("Invalid demonstration action schema")
        for key in ("image", "wrist_image"):
            if episode[key].shape != (count, 224, 224, 3) or episode[key].dtype != np.uint8:
                raise ValueError("Invalid demonstration camera schema")
        name = entry["file"]
        episodes[name] = episode
        splits[entry["split"]].extend((name, index) for index in range(0, count, stride))
        seeds[entry["split"]].add(entry["seed"])
    if seeds["train"] & seeds["validation"] or not all(splits.values()):
        raise ValueError("Training/validation episodes must be nonempty and disjoint")
    return episodes, splits, manifest


def phase_group(phase):
    if phase.startswith("recover"):
        return "recovery"
    return "contact" if phase == "seal" else phase


def phase_pools(episodes, refs):
    pools = {}
    for ref in refs:
        group = phase_group(episodes[ref[0]]["phases"][ref[1]])
        pools.setdefault(group, []).append(ref)
    return pools


def sample_refs(pools, count, rng):
    """Equal phase probabilities, then uniform anchors within each phase."""
    groups = sorted(pools)
    return [pools[group][int(rng.integers(len(pools[group])))]
            for group in (groups[int(i)] for i in rng.integers(len(groups), size=count))]


def evaluate_pickups(policy, parameters, seeds, seconds=20):
    """Learned actions only. Scene state is exclusively reset/evaluator data."""
    from panda_suction import PandaSuctionEnv, PandaSuctionDiagnostics
    from run_panda_card import verify_scene
    original = policy.model.weights
    policy.model.weights = {**original, **parameters}
    env = PandaSuctionEnv(seeds[0])
    results = []
    try:
        verify_scene(env)
        for seed in seeds:
            env.physics_observer = None
            env.scene_seed = seed
            obs = env.reset()
            for _ in range(4):
                obs, *_ = env.step([0.] * 6 + [-1.])
            diagnostics = PandaSuctionDiagnostics(env)
            env.physics_observer = diagnostics.physics_step
            policy.reset_seed(42)
            actions_log = []
            for start in range(0, round(seconds * 20), 5):
                inputs = env.observation_for_policy(obs, "Pick up the face-down card.")
                actions = policy.infer(inputs)["actions"]
                for action in actions[:min(5, round(seconds * 20) - start)]:
                    obs, *_ = env.step(action)
                    actions_log.append(action.tolist())
            result = {"seed":seed, **diagnostics.report(), "actions":actions_log}
            results.append(result)
            print(json.dumps({"development_seed":seed, "success":result["success"],
                              "clearance_mm":1000*result["maximum_card_clearance_m"],
                              "closest_mm":1000*result["minimum_tcp_to_card_surface_m"]}), flush=True)
    finally:
        env.close()
        policy.model.weights = original
    return results


def flow_matching_loss(parameters, frozen, cache, prefix_length, actions, noise, timestep):
    """OpenPI pi0.py compute_loss: x_t=t*noise+(1-t)*actions; v=noise-actions."""
    model = Pi05MLX({**frozen, **parameters}, eager=False)
    t = timestep[:, None, None]
    velocity = model.denoise(t * noise + (1 - t) * actions, timestep, cache, prefix_length)
    return mx.mean(mx.square(velocity - (noise - actions)))


def export_bundle(path, policy, parameters, provenance):
    """Write an independent inference bundle; never modify base checkpoint files."""
    path.mkdir(parents=True, exist_ok=False)
    weights = {**policy.model.weights, **parameters}
    manifest = {**policy.manifest, "weight_map": {}, "sha256": {}, "finetuning": provenance}
    shard, size, number = {}, 0, 0
    def flush():
        nonlocal shard, size, number
        if not shard:
            return
        filename = f"model-{number:04d}.safetensors"
        mx.save_safetensors(str(path / filename), shard)
        manifest["weight_map"].update({key: filename for key in shard})
        manifest["sha256"][filename] = sha256(path / filename)
        shard, size, number = {}, 0, number + 1
    for key in sorted(weights):
        value = weights[key]
        if size + value.nbytes > 512 * 1024**2:
            flush()
        shard[key] = value
        size += value.nbytes
    flush()
    for filename in ("paligemma_tokenizer.model", "norm_stats.json"):
        shutil.copyfile(Path(policy.metadata["checkpoint_path"]) / filename, path / filename)
        manifest["sha256"][filename] = sha256(path / filename)
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/pi05_libero_mlx"))
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2.5e-5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--cache-size", type=int, default=256)
    parser.add_argument("--validation-every", type=int, default=100)
    parser.add_argument("--validation-samples", type=int, default=16)
    parser.add_argument("--phase-balanced", action="store_true")
    parser.add_argument("--development-seeds", type=int, nargs="+")
    parser.add_argument("--rollout-every", type=int, default=400)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.stride, args.cache_size, args.validation_every, args.validation_samples, args.rollout_every) < 1:
        parser.error("Counts must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    episodes, splits, data_manifest = load_dataset(args.dataset, args.stride)
    dataset_seeds = {entry["seed"] for entry in data_manifest["episodes"]}
    if args.development_seeds and dataset_seeds.intersection(args.development_seeds):
        raise ValueError("Development rollout seeds must be separate from demonstration seeds")
    pools = {key:phase_pools(episodes, refs) for key, refs in splits.items()}
    policy = MLXPolicy(args.checkpoint)
    parameters = {key: value for key, value in policy.model.weights.items() if is_action_parameter(key)}
    frozen = {key: value for key, value in policy.model.weights.items() if key not in parameters}
    assert not any(key.startswith("PaliGemma/img/") for key in parameters)
    rng = np.random.default_rng(args.seed)
    cache_store = OrderedDict()
    prefix_length = None
    source_files = ["train_panda_mlx.py", "pi05_mlx.py", "collect_panda_demos.py",
                    "panda_suction.py", "panda_card_sim.py", "card_sim.py",
                    "run_panda_card.py", "episode_diagnostics.py"]
    for filename in source_files:
        target = args.output_dir / "source" / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(filename, target)
    provenance = {"method": "supervised flow matching; action expert only", "framework": "MLX Metal GPU",
                  "base_checkpoint_manifest_sha256": sha256(args.checkpoint / "manifest.json"),
                  "dataset_manifest_sha256": sha256(args.dataset / "manifest.json"),
                  "dataset": str(args.dataset), "training_seed": args.seed,
                  "normalization": "unchanged original LIBERO quantile statistics; identical action units",
                  "trainable_parameters": sum(value.size for value in parameters.values()),
                  "trainable_keys": sorted(parameters), "frozen_parameters": sum(value.size for value in frozen.values()),
                  "action_horizon": 10, "padded_action_dim": 32, "loss_dimensions": "all 32, matching upstream",
                  "time_distribution": "Beta(1.5,1)*0.999+0.001", "optimizer": "AdamW b1=.9 b2=.95 eps=1e-8 decay=1e-10; global norm clip=1",
                  "batch_size": args.batch_size, "learning_rate": args.learning_rate, "max_steps": args.steps,
                  "data_augmentation": "none; variation comes from recorded physical trajectories",
                  "phase_balanced": args.phase_balanced,
                  "phase_anchor_counts": {key:{phase:len(refs) for phase,refs in group.items()} for key,group in pools.items()},
                  "development_seeds": args.development_seeds,
                  "rollout_every": args.rollout_every,
                  "source_sha256": {file:sha256(file) for file in source_files},
                  "expert_provenance": data_manifest["source"], "training_observations_contain_object_pose": False}
    report = {"status": "training", "provenance": provenance, "device": policy.metadata["metal"],
              "samples": {key: len(value) for key, value in splits.items()}, "history": []}
    (args.output_dir / "training.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"trainable_parameters": provenance["trainable_parameters"], "frozen_parameters":provenance["frozen_parameters"], "device":report["device"]["device_name"], "samples":report["samples"]}), flush=True)

    def get_sample(ref):
        nonlocal prefix_length
        name, index = ref
        episode = episodes[name]
        if ref not in cache_store:
            inputs = {"observation/image":episode["image"][index], "observation/wrist_image":episode["wrist_image"][index],
                      "observation/state":episode["state"][index], "prompt":str(episode["prompt"])}
            images, tokens = policy.prepare(inputs)
            cache, length = policy.model.encode_prefix(images, tokens)
            mx.eval(cache)
            if prefix_length is not None and prefix_length != length:
                raise ValueError("This batched trainer requires the same prompt length for all samples")
            prefix_length = length
            cache_store[ref] = cache
            if len(cache_store) > args.cache_size:
                cache_store.popitem(last=False)
        cache_store.move_to_end(ref)
        actions = action_chunk(episode["actions"], index)
        normalized = (actions - policy.q01) / (policy.q99 - policy.q01 + 1e-6) * 2 - 1
        padded = np.pad(normalized, ((0, 0), (0, 25))).astype(np.float32)
        return cache_store[ref], padded

    def batch(refs, generator):
        samples = [get_sample(ref) for ref in refs]
        cache = [(mx.concatenate([s[0][i][0] for s in samples], axis=0),
                  mx.concatenate([s[0][i][1] for s in samples], axis=0)) for i in range(18)]
        actions = mx.array(np.stack([s[1] for s in samples]))
        noise = mx.array(generator.standard_normal(actions.shape).astype(np.float32))
        timestep = mx.array((generator.beta(1.5, 1., len(refs)) * .999 + .001).astype(np.float32))
        return cache, actions, noise, timestep

    def loss(params, cache, actions, noise, timestep):
        return flow_matching_loss(params, frozen, cache, prefix_length, actions, noise, timestep)

    value_and_grad = mx.value_and_grad(loss)
    optimizer = optim.AdamW(learning_rate=args.learning_rate, betas=(.9, .95), eps=1e-8, weight_decay=1e-10)
    val_rng = np.random.default_rng(12345)
    chosen = val_rng.choice(len(splits["validation"]), min(args.validation_samples, len(splits["validation"])), replace=False)
    val_refs = (sample_refs(pools["validation"], args.validation_samples, val_rng) if args.phase_balanced
                else [splits["validation"][int(i)] for i in chosen])
    def validate(params):
        generator = np.random.default_rng(54321)  # Fixed inputs/noise/times across checkpoints.
        losses = []
        for start in range(0, len(val_refs), args.batch_size):
            data = batch(val_refs[start:start + args.batch_size], generator)
            losses.append(float(loss(params, *data).item()))
        return float(np.mean(losses))

    best_loss = validate(parameters)
    report["initial_validation_loss"] = best_loss
    print(json.dumps({"step":0, "validation_loss":best_loss}), flush=True)
    best_parameters = dict(parameters)
    best_step = 0
    minimum_flow_loss, minimum_flow_step = best_loss, 0
    best_score = (-1, -float("inf"))
    report["development_evaluations"] = []
    began = time.monotonic()
    try:
        for step in range(1, args.steps + 1):
            tick = time.monotonic()
            refs = (sample_refs(pools["train"], args.batch_size, rng) if args.phase_balanced else
                    [splits["train"][int(i)] for i in rng.integers(0, len(splits["train"]), args.batch_size)])
            data = batch(refs, rng)
            value, gradients = value_and_grad(parameters, *data)
            gradients, norm = optim.clip_grad_norm(gradients, max_norm=1.)
            mx.eval(value, norm)
            if not math.isfinite(value.item()) or not math.isfinite(norm.item()) or norm.item() == 0:
                raise RuntimeError(f"Invalid loss/gradient at step {step}: {value.item()}, {norm.item()}")
            if step == 1:
                report["first_gradient_norms"] = {key:float(mx.sqrt(mx.sum(g * g)).item()) for key,g in gradients.items()}
                report["first_step_all_gradients_finite"] = all(bool(mx.all(mx.isfinite(g)).item()) for g in gradients.values())
                if not report["first_step_all_gradients_finite"]:
                    raise RuntimeError("Nonfinite parameter gradients")
            optimizer.learning_rate = args.learning_rate * min(1., step / min(20, args.steps))
            parameters = optimizer.apply_gradients(gradients, parameters)
            mx.eval(parameters, optimizer.state)
            row = {"step":step, "loss":float(value.item()), "gradient_norm":float(norm.item()),
                   "step_seconds":time.monotonic() - tick, "peak_memory_bytes":mx.get_peak_memory()}
            if step % args.validation_every == 0 or step == args.steps:
                row["validation_loss"] = validate(parameters)
                if not args.development_seeds and row["validation_loss"] < best_loss:
                    best_loss, best_step, best_parameters = row["validation_loss"], step, dict(parameters)
                if row["validation_loss"] < minimum_flow_loss:
                    minimum_flow_loss, minimum_flow_step = row["validation_loss"], step
                report["best_validation_step"] = minimum_flow_step
                report["best_validation_loss"] = minimum_flow_loss
            if args.development_seeds and (step % args.rollout_every == 0 or step == args.steps):
                # Preserve candidate before long simulator evaluation.
                candidate = args.output_dir / f"action_expert_step_{step:05d}.safetensors"
                mx.save_safetensors(str(candidate), parameters)
                print(json.dumps({"step":step, "status":"evaluating_learned_pickups"}), flush=True)
                outcomes = evaluate_pickups(policy, parameters, args.development_seeds)
                if "validation_loss" not in row:
                    row["validation_loss"] = validate(parameters)
                successes = sum(result["success"] for result in outcomes)
                score = (successes, -row["validation_loss"])
                report["development_evaluations"].append({"step":step, "successes":successes,
                                                          "validation_loss":row["validation_loss"], "episodes":outcomes})
                if score > best_score:
                    best_score = score
                    best_loss, best_step, best_parameters = row["validation_loss"], step, dict(parameters)
                report["selected_step"] = best_step
                report["selected_development_successes"] = best_score[0]
                report["selected_validation_loss"] = best_loss
            report["history"].append(row)
            if step == 1 or step % 10 == 0 or step == args.steps:
                print(json.dumps(row), flush=True)
                (args.output_dir / "training.json").write_text(json.dumps(report, indent=2))
        # Audit actual parameter updates, not just optimizer counters.
        report["updated_parameter_max_abs_delta"] = {key:float(mx.max(mx.abs(value - policy.model.weights[key])).item()) for key,value in best_parameters.items()}
        if best_step == 0 or not any(report["updated_parameter_max_abs_delta"].values()):
            raise RuntimeError("No updated checkpoint was selected; original checkpoint preserved")
        output = args.output_dir / "checkpoint"
        selection = ("maximum development pickup successes; fixed validation flow loss breaks ties"
                     if args.development_seeds else "minimum fixed held-out flow-matching loss")
        export_bundle(output, policy, best_parameters, {**provenance, "selected_step":best_step, "selection":selection})
        # Save/reload the changed tensors and verify exact bytes before evaluation.
        manifest = json.loads((output / "manifest.json").read_text())
        roundtrip = {}
        for filename in set(manifest["weight_map"][key] for key in best_parameters):
            roundtrip.update(mx.load(str(output / filename)))
        report["checkpoint_parameter_roundtrip_exact"] = all(bool(mx.array_equal(value, roundtrip[key]).item()) for key,value in best_parameters.items())
        if not report["checkpoint_parameter_roundtrip_exact"]:
            raise RuntimeError("Saved checkpoint differs from trained parameters")
        report.update(status="completed", checkpoint=str(output), wall_seconds=time.monotonic()-began,
                      peak_memory_bytes=mx.get_peak_memory(), frozen_weights_updated=False)
    except BaseException as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (args.output_dir / "training.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
