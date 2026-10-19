# Setup and reproduction

## Installed machine

Windows, 16 GB RAM, NVIDIA RTX 4070 Laptop GPU with 8,188 MiB VRAM. Conda environment: `C:\Users\heng\miniconda3\envs\simenv`, Python 3.11.17. Activate it with `conda activate simenv`, or invoke its `python.exe` directly.

The environment snapshot is [requirements-simenv.lock.txt](../requirements-simenv.lock.txt). Important versions: MuJoCo 3.3.7, NumPy 1.26.4, PyTorch 2.7.1+cu126, torchvision 0.22.1+cu126, Transformers 4.53.2 with OpenPI patches, JAX 0.5.3 (CPU), flax 0.10.2 and orbax 0.11.13. CUDA inference uses PyTorch; native Windows JAX remains the slow CPU fallback.

OpenPI is installed editable with `--no-deps`; its full dependency declaration includes unrelated training environments and Linux CUDA JAX packages. Install the pinned dependencies sequentially. An early overlapping pip installation left mixed NumPy files; removing/reinstalling NumPy 1.26.4 repaired that environment. No such repair is needed for the current installation.

## Rebuild

```powershell
.\setup_simenv.ps1
conda activate simenv
python build_scene.py
python validate_sim.py
python tools/audit_adapter.py
```

The bootstrap installs dependencies, checks out the source revisions below, applies the upstream Transformers patches, downloads the original public checkpoint and converts it for GPU inference. Original Transformers files are retained under `models/transformers-original`. The lock file is a snapshot containing local editable paths and a CUDA wheel; it is not a portable cross-platform lock.

## Run and record

```powershell
# Frozen policy with the corrected adapter, no RGB assistance:
python run_policy.py --backend cuda --seconds 20 --seed 42 --output-dir runs/my_unassisted_episode
# Vary the scene while holding the policy's CUDA sampling seed fixed:
python run_policy.py --backend cuda --seconds 20 --seed 0 --policy-noise-seed 42 --output-dir runs/my_scene0_noise42
# Two unassisted seeds, reusing one frozen model:
python tools/run_scenarios.py --seeds 0 1 --output-dir runs/my_seed_sweep
# Optional scene variants, without auxiliary control:
python tools/run_scenarios.py --seeds 0 1 --scenarios rotated left --output-dir runs/my_scenarios
# View a passive scene; this does not start a policy:
python view_scene.py
# RGB camera images and scripted data recorder:
python card_sim.py
python collect_demos.py
# Optional slow CPU path or remote OpenPI server:
python run_policy.py --backend cpu --chunks 3
python run_policy.py --server HOST --port 8000 --seconds 20
```

Use a fresh output directory for every experiment. Videos play at simulated time, with physics paused during inference. Runs save raw actions, observations, telemetry, intervention logs, evaluator diagnostics and MP4 videos. Future source snapshots also include the CUDA loader/model-construction source and scene/profile/runtime assets, with SHA-256 hashes, to avoid the incomplete historical provenance encountered in the seed audit. Weights are not copied into episode snapshots. `--image-transform auto` follows the profile; manual alternatives are `native`, `mirror_x`, `rotate_180`. The default profile is `suction_upright`; `legacy`, `libero`, `suction_refined` and `suction_rgb256` reproduce earlier ablations.

## Retained assets

| Component | Source/version | Local location |
| --- | --- | --- |
| OpenPI | `215abfb217dbac7d5f1273282331b9b1866c0479` | `vendor/openpi` |
| UR description | `d4042c1f5123c86645cc5dbd650b4f461f697ee9` | `vendor/ur_description` |
| Official checkpoint | `gs://openpi-assets/checkpoints/pi05_libero` | `models/pi05_libero` |
| Converted bf16 checkpoint | upstream layout conversion, no training | `models/pi05_libero_torch` |
| Tokenizer | official PaliGemma tokenizer | `models/cache/big_vision/paligemma_tokenizer.model` |
| Controller/camera reference | robosuite v1.4.1 | `vendor/reference` |

Original checkpoint files occupy about 12.4 GB; the converted bf16 version occupies another 6.7 GB. Conversion/loading avoid allocating two full models simultaneously. One embedding tensor exceeds the nominal shard size and has its own 1.05 GB shard. See `download_manifest.json`, `scene_manifest.json` and run reports for provenance.

The GPU loader validates all parameter devices as `cuda:0`. Compilation is disabled on this Windows installation; inference uses native PyTorch CUDA. Reproduction tools: `tools/validate_cuda.py`, `tools/audit_adapter.py`, `tools/probe_policy_failure.py`, `tools/summarize_episodes.py`. The former RGB helper and its audit/replay tools have been removed; their historical results remain in `runs/` and the archived notes.

`--policy-noise-seed` is supported for local CUDA inference; omitting it retains the prior behavior where scene and policy sampling use the same seed. `tools/diagnose_seeds.py` reproduces saved-input and scene/noise cross checks into a new reproduction directory; `tools/audit_policy_adaptation.py` runs the predeclared adapter/extra-seed audit described in [the seed diagnosis](seed-diagnosis.md). Use fresh output directories rather than overwriting recorded evidence.

On this hybrid laptop, create the first OpenGL context **before importing Torch** to match standalone rendering. `tools/run_scenarios.py` and the seed tools do this with `render_runtime.graphics_before_cuda()`. Otherwise importing Torch first can select NVIDIA rendering instead of Intel Arc Pro, changing RGB observations despite identical MuJoCo state. Policy inference stays on RTX 4070 in either case. Each new episode records `render_backend`; compare it along with source/configuration hashes. This ordering fixes the known adapter switch, but does not guarantee bit-exact rendered pixels throughout an episode.

`reference_scale_audit` is an experimental ablation profile with no upright guard or extra Cartesian norm clipping. It is not a replacement default or a reproduction of Franka controller dynamics. To compare the original JAX and converted CUDA policy on fixed saved inputs/noise, run `python tools/check_policy_parity.py --prepare-only` first, then `python tools/check_policy_parity.py`. Run the CPU reference after releasing the full GPU-model process to fit system RAM.

The expanded numerical comparison currently fails its declared tolerance on seed 0; seeds 1/42 pass. See the seed diagnosis before interpreting CUDA as numerically interchangeable with the original CPU model. `python tools/audit_attention_numerics.py` reproduces the saved-input attention arithmetic ablation without modifying the default runtime or checkpoint.
