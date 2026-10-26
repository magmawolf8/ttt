"""Native MLX inference for the pinned OpenPI pi05_libero checkpoint.

Implements SigLIP, the Gemma prefix, and the adaptive Gemma action expert using
the original Flax weight layouts. Equations follow OpenPI at revision
215abfb217dbac7d5f1273282331b9b1866c0479 (Apache-2.0; see THIRD_PARTY_NOTICES.md).
The runtime does not import Torch, JAX, or the community pi05-base port.
"""
import json
from importlib.metadata import version
import math
import os
from pathlib import Path
import sys
import time

# MLX 0.32 can use reduced-precision matrix units for float32 arrays on M5.
# Configure this before MLX initializes; the policy's precision contract is FP32.
if "mlx.core" in sys.modules and os.environ.get("MLX_ENABLE_TF32") != "0":
    raise RuntimeError("Set MLX_ENABLE_TF32=0 before importing mlx.core for pi0.5")
os.environ["MLX_ENABLE_TF32"] = "0"
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import sentencepiece

ROOT = Path(__file__).resolve().parent


def attention(q, k, v, scale, mask=None):
    # Keep attention in explicit FP32 operations, with query scaling before the
    # dot product to follow the upstream Flax operation order.
    scores = (q * scale) @ k.swapaxes(-1, -2)
    if mask is not None:
        scores = mx.where(mask, scores, -2.3819763e38)
    return mx.softmax(scores, axis=-1) @ v


def layer_norm(x, scale, bias):
    # Flax LayerNorm uses E[x²] - E[x]², clipped against roundoff.
    mean = mx.mean(x, axis=-1, keepdims=True)
    variance = mx.maximum(mx.mean(x * x, axis=-1, keepdims=True) - mean * mean, 0)
    return (x - mean) * mx.rsqrt(variance + 1e-6) * scale + bias


def rope(x, positions):
    radians = positions[:, :, None, None] / (
        10000 ** (mx.arange(x.shape[-1] // 2, dtype=mx.float32) * (2 / x.shape[-1]))
    )
    first, second = mx.split(x, 2, axis=-1)
    return mx.concatenate([first * mx.cos(radians) - second * mx.sin(radians),
                           second * mx.cos(radians) + first * mx.sin(radians)], axis=-1)


class Pi05MLX:
    def __init__(self, weights):
        self.weights = weights

    def w(self, name, layer=None):
        value = self.weights[name]
        return value if layer is None else value[layer]

    def dense(self, x, name, layer=None):
        return x @ self.w(name + "/kernel", layer) + self.w(name + "/bias", layer)

    def vision(self, images):
        p = "PaliGemma/img/"
        kernel = self.w(p + "embedding/kernel").transpose(3, 0, 1, 2)
        x = mx.conv2d(images, kernel, stride=14) + self.w(p + "embedding/bias")
        x = x.reshape(images.shape[0], 256, 1152) + self.w(p + "pos_embedding")
        block = p + "Transformer/encoderblock/"
        for i in range(27):
            y = layer_norm(x, self.w(block + "LayerNorm_0/scale", i),
                           self.w(block + "LayerNorm_0/bias", i))
            attn = block + "MultiHeadDotProductAttention_0/"
            q, k, v = [
                (y @ self.w(attn + name + "/kernel", i).reshape(1152, 1152)
                 + self.w(attn + name + "/bias", i).reshape(1152))
                .reshape(images.shape[0], 256, 16, 72).transpose(0, 2, 1, 3)
                for name in ("query", "key", "value")
            ]
            y = attention(q, k, v, scale=72**-0.5)
            y = y.transpose(0, 2, 1, 3).reshape(images.shape[0], 256, 1152)
            x = x + (y @ self.w(attn + "out/kernel", i).reshape(1152, 1152) + self.w(attn + "out/bias", i))
            y = layer_norm(x, self.w(block + "LayerNorm_1/scale", i),
                           self.w(block + "LayerNorm_1/bias", i))
            y = nn.gelu_approx(self.dense(y, block + "MlpBlock_0/Dense_0", i))
            x = x + self.dense(y, block + "MlpBlock_0/Dense_1", i)
            mx.eval(x)
        x = layer_norm(x, self.w(p + "Transformer/encoder_norm/scale"),
                       self.w(p + "Transformer/encoder_norm/bias"))
        return self.dense(x, p + "head")

    def norm(self, x, path, layer=None, condition=None):
        normed = x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-6)
        if condition is None:
            return normed * (1 + self.w(path + "/scale", layer)), None
        modulation = self.dense(condition, path + "/Dense_0", layer)[:, None, :]
        scale, shift, gate = mx.split(modulation, 3, axis=-1)
        return normed * (1 + scale) + shift, gate

    def transformer(self, x, positions, mask=None, cache=None, condition=None):
        suffix = "_1" if condition is not None else ""
        p = "PaliGemma/llm/layers/"
        caches = []
        batch, length, width = x.shape
        for i in range(18):
            y, gate = self.norm(x, p + "pre_attention_norm" + suffix, i, condition)
            qweight = self.w(p + "attn/q_einsum" + suffix + "/w", i)
            kvweight = self.w(p + "attn/kv_einsum" + suffix + "/w", i)
            q = (y @ qweight.transpose(1, 0, 2).reshape(width, 2048)).reshape(batch, length, 8, 256)
            k = (y @ kvweight[0].reshape(width, 256)).reshape(batch, length, 1, 256)
            v = (y @ kvweight[1].reshape(width, 256)).reshape(batch, length, 1, 256)
            q = rope(q, positions).transpose(0, 2, 1, 3)
            k = rope(k, positions).transpose(0, 2, 1, 3)
            v = v.transpose(0, 2, 1, 3)
            if cache is not None:
                k = mx.concatenate([cache[i][0], k], axis=2)
                v = mx.concatenate([cache[i][1], v], axis=2)
            caches.append((k, v))
            y = attention(q, k, v, scale=256**-0.5, mask=mask)
            y = y.transpose(0, 2, 1, 3).reshape(batch, length, 2048)
            y = y @ self.w(p + "attn/attn_vec_einsum" + suffix + "/w", i).reshape(2048, width)
            x = x + (y if gate is None else y * gate)
            y, gate = self.norm(x, p + "pre_ffw_norm" + suffix, i, condition)
            gating = self.w(p + "mlp" + suffix + "/gating_einsum", i)
            y = nn.gelu_approx(y @ gating[0]) * (y @ gating[1])
            y = y @ self.w(p + "mlp" + suffix + "/linear", i)
            x = x + (y if gate is None else y * gate)
            mx.eval(x, k, v)
        x, _ = self.norm(x, "PaliGemma/llm/final_norm" + suffix, condition=condition)
        return x, caches

    def encode_prefix(self, images, tokens):
        # Only valid tokens are included: the absent third camera and padded
        # language tokens are masked in OpenPI and cannot affect valid outputs.
        image_tokens = self.vision(images).reshape(1, -1, 2048)
        text = self.w("PaliGemma/llm/embedder/input_embedding")[tokens] * math.sqrt(2048)
        prefix = mx.concatenate([image_tokens, text], axis=1)
        positions = mx.arange(prefix.shape[1])[None, :]
        _, cache = self.transformer(prefix, positions)
        return cache, prefix.shape[1]

    def denoise(self, noisy_actions, timestep, cache, prefix_length):
        fraction = mx.linspace(0, 1, 512)
        period = 4e-3 * (4.0 / 4e-3) ** fraction
        phase = timestep[:, None] * (2 * math.pi / period)
        condition = mx.concatenate([mx.sin(phase), mx.cos(phase)], axis=-1)
        condition = nn.silu(self.dense(condition, "time_mlp_in"))
        condition = nn.silu(self.dense(condition, "time_mlp_out"))
        actions = self.dense(noisy_actions, "action_in_proj")
        positions = (mx.arange(actions.shape[1]) + prefix_length)[None, :]
        output, _ = self.transformer(actions, positions, cache=cache, condition=condition)
        return self.dense(output, "action_out_proj")

    def sample_actions(self, images, tokens, noise):
        cache, length = self.encode_prefix(images, tokens)
        actions = noise
        timestep = mx.array([1.0], dtype=mx.float32)
        for _ in range(10):
            actions = actions - 0.1 * self.denoise(actions, timestep, cache, length)
            timestep = timestep - 0.1
            mx.eval(actions)
        return actions


class MLXPolicy:
    def __init__(self, checkpoint=None, seed=42):
        if not mx.metal.is_available():
            raise RuntimeError("MLX GPU inference requires Apple Silicon with Metal")
        mx.set_default_device(mx.gpu)
        path = Path(checkpoint or ROOT / "models/pi05_libero_mlx")
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError("Prepare MLX weights: python tools/convert_pi05_mlx.py")
        self.manifest = json.loads(manifest_path.read_text())
        contract = {"format": "openpi-flax-mlx-v1", "action_horizon": 10,
                    "action_dim": 32, "num_steps": 10, "max_token_len": 200,
                    "discrete_state_input": False, "dtype": "float32", "quantized": False}
        if any(self.manifest.get(key) != value for key, value in contract.items()):
            raise ValueError("Incompatible MLX checkpoint manifest")
        start = time.monotonic()
        weights = {}
        for filename in dict.fromkeys(self.manifest["weight_map"].values()):
            weights.update(mx.load(str(path / filename)))
        if set(weights) != set(self.manifest["shapes"]):
            raise ValueError("Checkpoint tensor names do not match manifest")
        for key, value in weights.items():
            if list(value.shape) != self.manifest["shapes"][key] or value.dtype != mx.float32:
                raise ValueError(f"Checkpoint tensor shape/dtype mismatch: {key}")
        mx.eval(weights)
        self.model = Pi05MLX(weights)
        self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(path / "paligemma_tokenizer.model"))
        stats = json.loads((path / "norm_stats.json").read_text())["norm_stats"]["actions"]
        self.q01 = np.asarray(stats["q01"], dtype=np.float32)
        self.q99 = np.asarray(stats["q99"], dtype=np.float32)
        self.reset_seed(seed)
        self.metadata = {"engine": "mlx", "device": str(mx.default_device()),
                         "mlx_version": version("mlx"),
                         "mlx_enable_tf32": os.environ["MLX_ENABLE_TF32"],
                         "metal": mx.device_info(), "precision": "float32",
                         "checkpoint_path": str(path.resolve()),
                         "source": self.manifest["source"], "quantized": False,
                         "parameter_count": self.manifest["parameter_count"],
                         "rng": "numpy-PCG64-normal-float32", "seed": seed}
        print(f"MLX model ready in {time.monotonic() - start:.1f}s on {mx.default_device()}", flush=True)

    def reset_seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def prepare(self, observation):
        images = []
        for key in ("observation/image", "observation/wrist_image"):
            image = np.asarray(observation[key])
            if image.shape != (224, 224, 3) or image.dtype != np.uint8:
                raise ValueError(f"{key} must be 224x224 uint8 RGB")
            images.append(image.astype(np.float32) / 255.0 * 2.0 - 1.0)
        prompt = str(observation["prompt"]).strip().replace("_", " ").replace("\n", " ")
        tokens = self.tokenizer.encode(prompt, add_bos=True) + self.tokenizer.encode("\n")
        return mx.array(np.stack(images)), mx.array([tokens[:200]], dtype=mx.int32)

    def infer(self, observation, noise=None):
        images, tokens = self.prepare(observation)
        if noise is None:
            noise = self.rng.standard_normal((1, 10, 32)).astype(np.float32)
        noise = np.asarray(noise, dtype=np.float32)
        if noise.shape != (1, 10, 32) or not np.isfinite(noise).all():
            raise ValueError("Noise must be finite with shape (1, 10, 32)")
        self.last_noise = noise.copy()
        output = self.model.sample_actions(images, tokens, mx.array(noise))
        actions = np.asarray(output[0, :, :7])
        actions = (actions + 1.0) / 2.0 * (self.q99 - self.q01 + 1e-6) + self.q01
        if not np.isfinite(actions).all():
            raise RuntimeError("MLX generated nonfinite actions")
        return {"actions": actions}
