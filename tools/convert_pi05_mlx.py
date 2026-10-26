"""Export the original LIBERO Orbax arrays to a self-contained MLX bundle.

No tensor-layout conversion, quantization, or training is performed. The original
Flax tensor names/shapes are retained so the MLX implementation is auditable.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request

import jax
import numpy as np
import orbax.checkpoint as ocp
from flax.traverse_util import flatten_dict
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[1]
REVISION = "215abfb217dbac7d5f1273282331b9b1866c0479"


def restore_checkpoint(path):
    with ocp.PyTreeCheckpointer() as checkpointer:
        item = {"params": checkpointer.metadata(path)["params"]}
        tree = checkpointer.restore(path, ocp.args.PyTreeRestore(
            item=item, restore_args=jax.tree.map(
                lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray, dtype=np.float32), item
            )
        ))["params"]
    return {"/".join(key[:-1] if key[-1] == "value" else key): value
            for key, value in flatten_dict(tree).items()}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    source = ROOT / "models/pi05_libero"
    destination = ROOT / "models/pi05_libero_mlx"
    revision = subprocess.check_output(
        ["git", "-C", str(ROOT / "vendor/openpi"), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise RuntimeError(f"Expected pinned OpenPI {REVISION}, found {revision}")
    destination.mkdir(parents=True, exist_ok=True)
    print("Restoring original LIBERO checkpoint as float32...", flush=True)
    arrays = restore_checkpoint(source / "params")
    # Reject another architecture before producing a seemingly usable bundle.
    expected = {
        "PaliGemma/llm/embedder/input_embedding": (257152, 2048),
        "PaliGemma/img/embedding/kernel": (14, 14, 3, 1152),
        "PaliGemma/llm/layers/attn/q_einsum/w": (18, 8, 2048, 256),
        "action_in_proj/kernel": (32, 1024),
        "action_out_proj/kernel": (1024, 32),
    }
    for key, shape in expected.items():
        if arrays[key].shape != shape:
            raise ValueError(f"Unexpected shape {key}: {arrays[key].shape}")
    weight_map, hashes, shapes = {}, {}, {}
    shard, shard_bytes, shard_id = {}, 0, 0

    def flush():
        nonlocal shard, shard_bytes, shard_id
        if not shard:
            return
        filename = f"model-{shard_id:03d}.safetensors"
        path = destination / filename
        save_file(shard, str(path))
        hashes[filename] = sha256(path)
        weight_map.update({key: filename for key in shard})
        print(f"Saved {filename}: {shard_bytes / 1e9:.2f} GB", flush=True)
        shard, shard_bytes = {}, 0
        shard_id += 1

    for key in sorted(arrays):
        value = arrays[key]
        if shard_bytes + value.nbytes > 512 * 1024**2:
            flush()
        shard[key] = np.ascontiguousarray(value)
        shapes[key] = list(value.shape)
        shard_bytes += value.nbytes
    flush()
    stats = destination / "norm_stats.json"
    shutil.copyfile(source / "assets/physical-intelligence/libero/norm_stats.json", stats)
    tokenizer = destination / "paligemma_tokenizer.model"
    if not tokenizer.exists():
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model", tokenizer
        )
    hashes[stats.name] = sha256(stats)
    hashes[tokenizer.name] = sha256(tokenizer)
    manifest = {
        "format": "openpi-flax-mlx-v1", "source": "gs://openpi-assets/checkpoints/pi05_libero",
        "openpi_revision": revision, "dtype": "float32", "quantized": False,
        "action_horizon": 10, "action_dim": 32, "num_steps": 10,
        "max_token_len": 200, "discrete_state_input": False,
        "weight_map": weight_map, "shapes": shapes, "sha256": hashes,
        "parameter_count": sum(value.size for value in arrays.values()),
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"MLX bundle ready: {destination}", flush=True)


if __name__ == "__main__":
    main()
