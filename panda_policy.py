"""Backend selection and reproducible sampling for the Panda/card runner.

Neither GPU framework is imported until its backend is selected. Call the
factory after creating the simulator's first graphics context.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


class SeededPolicy:
    """Supply the same saved PCG64 noise to either backend's native inference."""

    def __init__(self, policy, metadata, seed):
        self.policy = policy
        self.metadata = {**metadata, "rng": "numpy-PCG64-normal-float32", "seed": seed}
        self.last_noise = None
        self.reset_seed(seed)

    def reset_seed(self, seed):
        self.rng = np.random.default_rng(seed)
        self.metadata["seed"] = seed

    def infer(self, observation, noise=None):
        if noise is None:
            noise = self.rng.standard_normal((1, 10, 32)).astype(np.float32)
        noise = np.asarray(noise, dtype=np.float32)
        if noise.shape != (1, 10, 32) or not np.isfinite(noise).all():
            raise ValueError("Noise must be finite with shape (1, 10, 32)")
        self.last_noise = noise.copy()
        result = self.policy.infer(observation, noise=noise)
        actions = np.asarray(result["actions"])
        if actions.shape != (10, 7) or not np.isfinite(actions).all():
            raise ValueError("Policy must return finite actions with shape (10, 7)")
        return result


def load_panda_policy(backend, seed=42, checkpoint=None):
    if checkpoint is not None and backend != "mlx":
        raise ValueError("Custom checkpoints currently require --backend mlx; CUDA needs a matching PyTorch export")
    if backend == "mlx":
        from pi05_mlx import MLXPolicy
        policy = MLXPolicy(checkpoint=checkpoint, seed=seed)
        metadata = dict(policy.metadata)
        manifest_path = Path(metadata["checkpoint_path"]) / "manifest.json"
    elif backend == "cuda":
        import torch
        from pi05_torch import require_cuda
        require_cuda()
        # JAX provides CPU preprocessing / Orbax conversion only in this path.
        # It must not reserve CUDA memory needed by Torch and MuJoCo.
        import os
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        from pi05_policy import load_policy
        policy = load_policy(backend="cuda")
        manifest_path = ROOT / "models/pi05_libero_torch/model.safetensors.index.json"
        manifest = json.loads(manifest_path.read_text())
        metadata = {"engine": "pytorch-cuda", "device": str(torch.cuda.current_device()),
                    "gpu": torch.cuda.get_device_name(), "torch_version": torch.__version__,
                    "cuda_version": torch.version.cuda,
                    "compute_capability": list(torch.cuda.get_device_capability()),
                    "precision": "bfloat16 with OpenPI-selected float32 embeddings, norms and action projections",
                    "source": manifest["source"], "quantized": False,
                    "checkpoint_path": str(manifest_path.parent)}
    else:
        raise ValueError(f"Unknown Panda policy backend: {backend}")
    metadata["checkpoint_manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return SeededPolicy(policy, metadata, seed)


def peak_memory_bytes(backend):
    if backend == "mlx":
        import mlx.core as mx
        return mx.get_peak_memory()
    if backend == "cuda":
        import torch
        return torch.cuda.max_memory_allocated()
    raise ValueError(f"Unknown Panda policy backend: {backend}")
