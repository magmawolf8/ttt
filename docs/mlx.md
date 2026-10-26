# Apple Silicon MLX inference

The local `mlx` backend runs the original `pi05_libero` weights on Apple's GPU
and controls the existing MuJoCo scene. `pi05_mlx.py` implements the pinned
OpenPI model directly in MLX; it does not depend on the community pi05-base port.
MuJoCo physics and the action adapter are unchanged.

For successful pretrained grasp-and-place with the checkpoint's native Panda
arm and parallel gripper, see [LIBERO on MLX](libero-mlx.md) and run
`.venv-mlx/bin/mjpython run_libero_mlx.py --viewer`. The UR7e card results below
remain a separate, unsuccessful transfer experiment.

## Verified local result — 25 October 2026

Tested on an Apple M5 Max with 128 GB unified memory and MLX 0.32.4. The final
20-second simulated episode (`runs/mlx_demo_42`) completed in 56.0 wall seconds:
80 policy calls, 400 commands/video frames and 10,000 physics steps. Median
inference latency after the first call was 0.647 seconds per ten-action chunk;
peak MLX allocation was 14.35 GB. A live viewer episode also completed.

All eight simulator checks passed. Four fixed-input comparisons across the smoke
and live episodes passed the predeclared thresholds and had identical gripper
signs; the largest relative action error was 0.0224%. Every final episode command
was checked against its saved neural action, and suction matched the gripper sign.
This confirms functioning local inference/control, not pickup competence: the
final episode failed on approach/alignment and did not attach the card.

The [validation record](mlx-validation.json) includes provenance and the failed
early reduced-precision comparisons. Full recordings and source snapshots remain
in the ignored `runs/` directories. Watch `runs/mlx_demo_42/episode_dashboard.mp4`
or `policy_rollout.mp4` locally.

## Run in the prepared environment

```bash
# Recorded episode, no separate viewer window:
.venv-mlx/bin/python run_policy.py --backend mlx --seconds 20 --seed 42 --output-dir runs/my_mlx_episode

# Live viewer as well as the recordings (macOS requires mjpython):
.venv-mlx/bin/mjpython run_policy.py --backend mlx --viewer --seconds 20 --seed 42 --output-dir runs/my_mlx_live
```

Use a fresh output directory. Each run saves the same observations, raw actions,
bounded commands, telemetry and videos as the other backends, plus every sampled
noise tensor, the MLX device/precision and checkpoint manifest. Inference pauses
simulation time; the viewer may remain stationary while a new chunk is computed.

## Reconstruct

Requires Apple Silicon, a supported macOS version for MLX 0.32.4, Python 3.12,
Git and enough memory for unquantized inference. This implementation uses FP32
weights (roughly 13.4 GB); original plus exported checkpoints take roughly 26 GB
on disk. Additional memory is needed for activations and checkpoint conversion.
It has not been optimized for small-memory Macs.

```bash
bash setup_mlx.sh /path/to/python3.12
.venv-mlx/bin/python validate_sim.py --output-dir runs/my_mlx_sim_validation
```

Setup creates `.venv-mlx`, installs `requirements-mlx.txt`, retrieves the pinned
robot and OpenPI sources, downloads the original checkpoint and exports its
arrays to `models/pi05_libero_mlx`. The export retains Flax tensor names/layouts
and FP32 values. It includes the original normalizer and PaliGemma tokenizer;
there is no fine-tuning or quantization. JAX/Orbax are used only for export and
reference validation, never by the MLX inference runtime.

## Preserved contracts and differences

- External and wrist images remain 224 × 224 RGB with the profile's orientation.
- The tokenizer, seven-dimensional action unnormalization, ten-action horizon
  and ten Euler denoising steps follow the pinned LIBERO configuration.
- State conditioning remains disabled. Card pose and attachment feedback never
  enter the model.
- The fully masked third camera and padded language tokens are omitted from MLX
  computation. Valid tokens retain the same positions and attention visibility.
- MLX uses FP32 rather than the historical bf16 CUDA/JAX path. Its NumPy PCG64
  sampling generator differs from Torch's generator, so equal numeric seeds do
  not imply equal noise across backends. Use saved noise for comparisons.
- `pi05_mlx.py` sets `MLX_ENABLE_TF32=0` before importing MLX, and uses explicit
  FP32 attention. On M5, float32 arrays alone do not guarantee full-precision
  matrix operations. If embedding this backend in another application, set that
  environment variable before any `mlx.core` import. The backend rejects an
  earlier MLX import without this setting. See [MLX numerical precision](https://ml-explore.github.io/mlx/build/html/usage/precision.html).
- The default execution still replans after five commands and applies the same
  bounds and suction rules. Changing inference engines does not establish reliable
  pickup or repair the policy's embodiment mismatch.

## Independent numerical validation

Install the reference-only dependencies, then compare saved inputs/noise with the
original checkpoint and upstream JAX FP32 model:

```bash
.venv-mlx/bin/python -m pip install -r requirements-mlx-reference.txt
.venv-mlx/bin/python tools/check_mlx_parity.py runs/my_mlx_episode --chunks 0 1
```

This check uses upstream preprocessing, including the masked camera and padded
tokens, and writes `mlx_parity.json`. Its predeclared thresholds are relative L2
below 1% and maximum absolute unnormalized action error below 0.01. It also records
gripper sign agreement and translation error. The first CPU JAX compilation can
be slow. Passing these fixed-input checks does not establish equivalence of entire
episodes or equivalence to the historical bf16 CUDA results.
