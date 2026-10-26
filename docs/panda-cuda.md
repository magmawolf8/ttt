# Panda suction on NVIDIA / CUDA

The shared `run_panda_card.py` supports `--backend cuda` and `--backend mlx`.
Both use the same Panda scene, seven-value action contract and original suction
physics. CUDA loads the existing OpenPI PyTorch conversion of the official
`pi05_libero` checkpoint. No training, new action head, MLX installation, object
pose input or scripted pickup is involved.

## Fresh NVIDIA computer

Install Git, Python **3.12**, and a working NVIDIA driver. Run from this repo:

```bash
# Linux
python3.12 setup_panda_cuda.py
.venv-cuda/bin/python run_panda_card.py --backend cuda --tool suction --seeds 44 --policy-noise-seed 42
```

```powershell
# Windows / PowerShell
py -3.12 setup_panda_cuda.py
.\.venv-cuda\Scripts\python.exe run_panda_card.py --backend cuda --tool suction --seeds 44 --policy-noise-seed 42
```

Add `--viewer` for a live window on a desktop with a display. On a headless Linux
GPU server, run without `--viewer`; robosuite uses EGL for offscreen rendering.
The host must provide working OpenGL/EGL drivers as well as CUDA. If running in
a container, expose NVIDIA graphics capabilities in addition to compute.
Videos and telemetry are recorded without the viewer.

The setup creates `.venv-cuda`, retrieves pinned OpenPI/UR/LIBERO source,
installs PyTorch 2.7.1 and Transformers 4.53.2, applies the required upstream
Transformers patches, downloads the official weights and converts their layout
to PyTorch. It also caches the tokenizer. JAX is CPU-only here; neural inference
runs on CUDA. Original and converted weights occupy roughly 12.4 GB and 6.7 GB,
with additional space needed for dependencies and recordings.

The default wheel channel is `cu128`; `--cuda-wheel cu126` selects the original
CUDA 12.6 channel. Both channels are published for PyTorch 2.7.1 on
[Linux and Windows](https://pytorch.org/get-started/previous-versions/#v271).
Use a driver compatible with the selected channel. This runtime requires native
bfloat16 support (Ampere or newer). All model parameters stay on the GPU;
allow memory beyond the approximately 6.7 GB of weights for inference and rendering.
The original UR7e CUDA implementation ran on an 8 GB RTX 4070 Laptop GPU, but
the Panda configuration's peak NVIDIA memory use has not yet been measured.

`--dry-run` prints setup commands without installing or downloading anything.
`--skip-weights` prepares dependencies only; rerun without it before inference.
Existing dependency checkouts at unexpected revisions are preserved and reported
as errors instead of being reset automatically.

## Existing Windows simenv

The original `setup_simenv.ps1` remains the UR7e bootstrap. For the Panda runner,
prefer the separate Python 3.12 environment above. The legacy Windows package
snapshot is not used as a portable lock file.

## Evaluation and reproduction

```bash
# Three placements, same sampling seed, 20 simulated seconds per placement:
.venv-cuda/bin/python run_panda_card.py --backend cuda --seeds 42 43 44 --policy-noise-seed 42 --output-dir runs/my_cuda_sweep

# Explicitly scripted physical-adapter validation (not learned-policy evidence):
.venv-cuda/bin/python tools/check_panda_suction.py --output-dir runs/my_cuda_mechanics

# CPU contract checks, no weights required:
.venv-cuda/bin/python -m unittest discover -s tests -v
```

Use the Windows interpreter path from above in PowerShell. `--tool parallel`
selects the physical Panda fingers. `run_panda_card_mlx.py` remains a compatible
Mac entry point; new commands can use `run_panda_card.py --backend mlx`.

The runner renders before importing Torch to preserve graphics-adapter selection
on hybrid laptops. Reports identify the selected backend, graphics device,
Torch/CUDA or MLX version, checkpoint manifest hash, saved observations, raw
actions, sampling noise and peak framework memory. CUDA and MLX receive identical
NumPy PCG64 noise for a given seed. Each action chunk has shape `(10, 7)`;
the first six values go directly to Panda OSC_POSE, and output 6 controls vacuum
by sign. Physics and success scoring are shared.

**Validation limit:** the CUDA adapter and setup were prepared on a Mac without
an NVIDIA GPU. CPU contract checks exercise dispatch, deterministic noise,
action validation and failure handling; they are not a CUDA inference test.
The shared runner's full MLX regression reproduced the successful seed-44 pickup.
All 80 saved observation, noise and action chunks were bit-identical to the
preceding successful MLX run. See [the portable validation summary](panda-cuda-validation.json).
The earlier **1/3 success result applies to MLX only**. CUDA's bfloat16 arithmetic
and different rendering hardware can change trajectories; a CUDA pickup or
bit-identical cross-backend behavior is not established. Run the three-placement
sweep on the target GPU and inspect its `results.json`.
