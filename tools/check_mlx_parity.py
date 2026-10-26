"""Compare saved MuJoCo observations/noise with the pinned JAX FP32 model.

Uses the original Orbax weights and upstream preprocessing. This is a port
correctness check against FP32, not equivalence to the historical bf16 CUDA runs.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "vendor/openpi/src"),
                str(ROOT / "vendor/openpi/packages/openpi-client/src")]

import jax
import jax.numpy as jnp
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", type=Path)
    parser.add_argument("--chunks", type=int, nargs="+", default=[0])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    from openpi import transforms
    from openpi.models.model import Observation, restore_params
    from openpi.models.pi0_config import Pi0Config
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.policies.libero_policy import LiberoInputs, LiberoOutputs
    from openpi.shared import normalize, nnx_utils
    import sentencepiece

    config = Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False,
                      dtype="float32", pytorch_compile_mode=None)
    print("Loading upstream JAX FP32 reference from original Orbax checkpoint", flush=True)
    params = restore_params(ROOT / "models/pi05_libero/params", dtype=jnp.float32)
    model = config.load(params)
    del params
    gc.collect()
    infer = nnx_utils.module_jit(model.sample_actions)
    # Same upstream tokenizer class, with its already downloaded model injected
    # to avoid a redundant network/cache download during validation.
    tokenizer = PaligemmaTokenizer.__new__(PaligemmaTokenizer)
    tokenizer._max_len = config.max_token_len
    tokenizer._tokenizer = sentencepiece.SentencePieceProcessor(
        model_file=str(ROOT / "models/pi05_libero_mlx/paligemma_tokenizer.model")
    )
    stats = normalize.load(ROOT / "models/pi05_libero/assets/physical-intelligence/libero")
    prepare = transforms.compose([
        LiberoInputs(model_type=config.model_type),
        transforms.Normalize(stats, use_quantiles=True),
        transforms.ResizeImages(224, 224),
        transforms.TokenizePrompt(tokenizer, discrete_state_input=False),
        transforms.PadStatesAndActions(32),
    ])
    finish = transforms.compose([transforms.Unnormalize(stats, use_quantiles=True), LiberoOutputs()])
    report = {"reference": "pinned OpenPI JAX float32, original Orbax weights",
              "candidate": "MLX float32", "episode": str(args.episode),
              "thresholds": {"relative_l2": 0.01, "max_absolute_action": 0.01}, "checks": []}
    report["openpi_revision"] = subprocess.check_output(
        ["git", "-C", str(ROOT / "vendor/openpi"), "rev-parse", "HEAD"], text=True
    ).strip()
    episode_report = json.loads((args.episode / "results.json").read_text())
    report["candidate_source_sha256"] = episode_report["source_sha256"]["pi05_mlx.py"]
    report["checkpoint_manifest_sha256"] = hashlib.sha256(
        (ROOT / "models/pi05_libero_mlx/manifest.json").read_bytes()
    ).hexdigest()
    for chunk in args.chunks:
        with np.load(args.episode / f"observation_{chunk:03}.npz") as saved:
            observation = dict(saved)
        observation["prompt"] = str(observation["prompt"])
        inputs = prepare(observation)
        inputs = jax.tree.map(lambda value: jnp.asarray(value)[None], inputs)
        obs = Observation.from_dict(inputs)
        noise = np.load(args.episode / f"noise_{chunk:03}.npy")
        start = time.monotonic()
        print(f"Reference chunk {chunk}: compiling/executing...", flush=True)
        prediction = infer(jax.random.key(0), obs, noise=jnp.asarray(noise), num_steps=10)
        prediction.block_until_ready()
        actions = finish({"actions": np.asarray(prediction[0]), "state": np.asarray(obs.state[0])})["actions"]
        candidate = np.load(args.episode / f"actions_{chunk:03}.npy")
        error = actions - candidate
        relative = float(np.linalg.norm(error) / max(float(np.linalg.norm(actions)), 1e-12))
        maximum = float(np.max(np.abs(error)))
        signs_match = bool(np.array_equal(actions[:, 6] > 0, candidate[:, 6] > 0))
        check = {"chunk": chunk, "relative_l2": relative, "max_absolute_action": maximum,
                 "max_xyz_difference_m_before_bounds": float(np.max(np.abs(error[:, :3])) * .05),
                 "gripper_signs_match": signs_match,
                 "reference_seconds": time.monotonic() - start,
                 "passed": relative < .01 and maximum < .01 and signs_match}
        report["checks"].append(check)
        np.save(args.episode / f"jax_fp32_actions_{chunk:03}.npy", actions)
        print(json.dumps(check), flush=True)
    report["passed"] = all(check["passed"] for check in report["checks"])
    output = args.output or args.episode / "mlx_parity.json"
    output.write_text(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
