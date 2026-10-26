# Pretrained pi0.5 with its native Panda robot

`run_libero_mlx.py` runs the official, already-trained `pi05_libero` checkpoint
with one Franka Panda arm and its parallel-jaw gripper in native LIBERO/MuJoCo.
The policy runs on the Apple GPU through MLX. No fine-tuning, scripted grasp,
visual servo, or object-pose input is used.

## Why switch from the UR7e suction scene?

The checkpoint is trained for LIBERO's cameras, robot, controller, gripper and
tasks. Mapping its gripper command to a vacuum switch does not teach suction
manipulation. In the recorded UR7e MLX run (`runs/mlx_demo_42`), the cup never
contacted the card and stayed at least 190.5 mm away. This was an approach
failure, before any attachment decision. Separate scripted checks established
that the suction mechanism can pick up and hold the card.

Two arms are **not** required. The upstream LIBERO adapter returns seven values:
three translation deltas, three rotation deltas and one gripper command. It
discards the other 25 dimensions of the model's padded 32-value action vector.
The absent second wrist camera is masked. See the pinned
[LIBERO adapter](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/policies/libero_policy.py)
and [evaluation client](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/libero/main.py).

This is a native bowl grasp-and-place baseline. It does **not** establish that
the model can pick up the custom thin card. The original card simulator remains
available via `run_policy.py`; its learned pickup problem is still unresolved.
No task-specific training was started before switching to this baseline.

## Run

In the prepared environment:

```bash
# Native live viewer plus recorded video; output directory is timestamped.
.venv-mlx/bin/mjpython run_libero_mlx.py --viewer

# Three distinct initial states, without a viewer:
.venv-mlx/bin/python run_libero_mlx.py --trials 3 --output-dir runs/panda_three_trials
```

The default task is LIBERO-Spatial task 0: pick up the black bowl between the
plate and ramekin and place it on the plate. `--suite`, `--task`, `--init-start`,
`--trials`, `--seed`, and `--policy-noise-seed` select the experiment. State and
sampling seeds are distinct. The sampling seed increases by one per trial.
Output directories must be new, preventing accidental replacement of evidence.

The runner uses upstream initial states, ten initial settling commands, 256-pixel
camera renders rotated 180 degrees and resized to 224, the task's original
language instruction, and five commands per inference chunk. Seven-value model
actions go directly into LIBERO's OSC_POSE controller; there is no UR7e action
adapter or vacuum conversion. Physics runs at LIBERO's native settings and
pauses during inference; playback is at the 20 Hz control rate.

Success is the environment's native task predicate, checked after every command.
An already-successful initial state is rejected. Each episode saves policy
observations, model action chunks, sampled noise, executed-action telemetry,
a two-camera video and a final frame. Object positions in telemetry are
evaluator-only and never enter `policy_observation()` or model inference.

## Setup and macOS compatibility

After the [MLX setup](mlx.md):

```bash
bash setup_libero_mlx.sh
```

This installs `requirements-libero-mlx.txt` and checks out LIBERO at
`f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`, the submodule revision pinned by our
OpenPI checkout. Torch is needed only because LIBERO serializes its initial
simulation states in that format; Torch does not run the policy. The model
weights, tokenizer and normalizer are the same exported checkpoint used by the
UR7e MLX run.

Robosuite 1.4.1's offscreen context falls back to a hidden GLFW window on macOS,
which aborts under `mjpython` because AppKit requires window creation on the main
thread. `configure_renderer()` replaces only this context constructor with
MuJoCo's native CGL context. Camera geometry, image orientation, controllers and
physics are unchanged. This compatibility hook is applied before constructing
the environment and recorded in run metadata. LIBERO configuration and caches
are kept under the repository's ignored `runs/libero_config`, not the home folder.

## Local evidence

The first complete native trial (`runs/libero_mlx_panda_01`, initial state 0,
sampling seed 42) passed in 112 commands / 5.6 simulated seconds. The target bowl
rose 117.3 mm above its initial center height, then was placed on the plate.
All 112 executed actions exactly matched their saved neural outputs. Median
inference was 0.509 seconds per chunk; peak MLX allocation was 14.36 GB.

Two additional initial states (1 and 2; sampling seeds 43 and 44) also passed
with the live viewer and native CGL rendering (`runs/libero_mlx_panda_cgl`).
The completed total is **3/3 successful trials**. Every executed command in all
three trials matched the saved model output, and the target bowl rose more than
50 mm in each trial. No object poses were present in any saved policy input.

The first viewer attempt (`runs/libero_mlx_panda_02.log`) aborted during renderer
creation, before model control. Its incomplete `results.json` is not a policy
failure or a completed trial. The CGL compatibility hook addresses this crash.

These are local smoke trials on an M5 Max with 128 GB RAM, not a reproduction of
the full LIBERO benchmark. Results and source hashes are summarized in
`libero-mlx-validation.json`; complete logs, videos and inputs remain under `runs/`.
