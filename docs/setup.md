# Setup and reproduction

## Existing laptop environment

The working installation is Windows with 16 GB RAM and an RTX 4070 Laptop GPU (8,188 MiB VRAM). Use the `simenv` conda environment, normally at `%USERPROFILE%\miniconda3\envs\simenv`.

| Dependency | Recorded version |
| --- | --- |
| Python | 3.11.17 |
| MuJoCo | 3.3.7 |
| PyTorch | 2.7.1+cu126 |
| Transformers | 4.53.2 with pinned OpenPI patches |
| JAX | 0.5.3, CPU |
| NumPy | 1.26.4 |

The complete machine snapshot is `requirements-simenv.lock.txt` at repository root. It includes editable-source references and a CUDA wheel, so treat it as a reconstruction aid rather than a portable environment lock.

## Run an episode

From the repository root:

```powershell
conda activate simenv
python run_policy.py --backend cuda --seconds 20 --seed 42 --output-dir runs/my_episode
python tools/run_scenarios.py --seeds 0 1 --output-dir runs/my_seed_sweep
```

Use a fresh output directory for each experiment. Runs save observations, raw actions, telemetry, physics diagnostics, source/configuration snapshots and MP4 videos. A 20-second episode executes 400 commands and 80 policy calls. Physics pauses during inference; video follows simulated time.

Scene and policy sampling seeds are coupled by default. Hold sampling fixed to compare placements:

```powershell
python run_policy.py --backend cuda --seconds 20 --seed 0 --policy-noise-seed 42 --output-dir runs/my_scene0_noise42
python tools/run_scenarios.py --seeds 0 1 --scenarios rotated left --output-dir runs/my_scenarios
```

`--policy-noise-seed` applies only to local CUDA inference. The default adapter is `suction_upright`; `--image-transform auto` uses its image convention. Other profiles and manual transforms are development ablations, not validated improvements.

## Reconstruct the installation

The downloaded `models/`, `vendor/`, `datasets/` and complete `runs/` directories are not committed. With conda and Git installed, run:

```powershell
.\setup_simenv.ps1
conda activate simenv
python validate_sim.py
python tools/audit_adapter.py
```

The setup script creates or reuses `simenv`, installs pinned dependencies, checks out source revisions, applies the upstream Transformers patches, builds the scene, downloads the original checkpoint and converts it for GPU inference. The original and converted checkpoints occupy approximately 12.4 GB and 6.7 GB. Conversion is layout/precision preparation, not training.

| Asset | Pinned source |
| --- | --- |
| OpenPI | `215abfb217dbac7d5f1273282331b9b1866c0479` |
| UR description | `d4042c1f5123c86645cc5dbd650b4f461f697ee9` |
| Checkpoint | `gs://openpi-assets/checkpoints/pi05_libero` |
| Controller/camera reference used in audits | robosuite v1.4.1 |

`vendor/reference` contains development reference material in the original workspace; it is not populated by the bootstrap and is not needed by ordinary policy episodes. Some diagnostic tools also require the saved historical observations named in their source; the small documentation evidence bundle does not replace complete run outputs.

## Graphics and inference reproducibility

On this hybrid laptop, create the first OpenGL context **before importing Torch**. The standalone runner does so naturally; the batch and seed-audit tools use `render_runtime.graphics_before_cuda()`. Reversing the order can select NVIDIA instead of Intel Arc Pro rendering, changing the camera pixels. Policy inference remains on RTX 4070 in either case.

New reports record the GL vendor/renderer/version, model device and source/configuration hashes. Compare those along with scene/noise seeds. Matching them does not guarantee bit-exact pixels for a whole episode.

The CUDA loader puts all parameters on `cuda:0` with no CPU weight offload; compilation is disabled. Fixed-input CUDA/JAX agreement is not uniform: the expanded seed-0 check failed its declared tolerance. See [results](results.md) before interpreting the two backends as numerically equivalent.

## Inspection and diagnostics

```powershell
# Passive scene viewer and independent scripted checks:
python view_scene.py
python validate_sim.py

# Scripted demonstrations use card ground truth and are labelled separately:
python collect_demos.py

# Slow CPU fallback or an existing remote OpenPI server:
python run_policy.py --backend cpu --chunks 3
python run_policy.py --server HOST --port 8000 --seconds 20

# Reproduce the development audit; requires its saved historical inputs:
python tools/audit_policy_adaptation.py --output-dir runs/my_adapter_audit
```

For original CPU-versus-CUDA checks, `tools/check_policy_parity.py --prepare-only` saves exact CUDA noise before `tools/check_policy_parity.py` runs the CPU reference. Release other full model processes first to fit system RAM. `tools/audit_attention_numerics.py` changes attention arithmetic only inside its diagnostic process.

The [experiment archive](archive/seed-audit.md) records complete protocols and limitations. Commands in the [hybrid archive](archive/hybrid-control.md) describe removed code and are historical only.
