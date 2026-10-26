# UR7e card manipulation in MuJoCo

A prototype toward a robot arm that manipulates board-game pieces: a UR7e suction tool, two RGB cameras, and frozen OpenPI pi0.5 inference on an RTX 4070 Laptop GPU. The current instruction is `Pick up the face-down card.`

The original UR7e simulation-to-policy pipeline runs, but none of its 14 historical controlled development episodes met the full hold criterion. The adapter is experimental; RGB centering and scripted lifting have been removed. A newer Panda + original suction-cup variant achieved one pretrained pickup in three tested placements on MLX (see below). Reliable pickup across placements remains unresolved. See the [UR7e results](docs/results.md) and [detailed seed audit](docs/archive/seed-audit.md).

## Run

For **NVIDIA / CUDA with the Panda suction cup**, run `python3.12 setup_panda_cuda.py`
on Linux or `py -3.12 setup_panda_cuda.py` on Windows, then use the created
environment to run `run_panda_card.py --backend cuda --tool suction --seeds 44`.
See [NVIDIA setup and validation limits](docs/panda-cuda.md). The CUDA path is
implemented; the new Panda suction configuration still needs testing on NVIDIA hardware.

For Apple Silicon, see [MLX setup and inference](docs/mlx.md). In a prepared
environment: `.venv-mlx/bin/mjpython run_policy.py --backend mlx --viewer --seconds 20 --seed 42 --output-dir runs/my_mlx_episode`.

For the checkpoint's native **Franka Panda + parallel gripper**, use
`.venv-mlx/bin/mjpython run_libero_mlx.py --viewer`. This learned policy has
completed a native bowl grasp-and-place task on MLX. See
[Panda setup and evidence](docs/libero-mlx.md). This uses native LIBERO tasks;
the custom card results depend on the robot/tool configuration described below.

The Panda in the **original card scene** now defaults to the original suction
cup, with unchanged pretrained weights:
`.venv-mlx/bin/mjpython run_panda_card_mlx.py --tool suction --viewer --seeds 44 --policy-noise-seed 42`.
See [the suction adapter and validation](docs/panda-suction-mlx.md).
This setup achieved one pretrained pickup in three tested placements; reliability
across placements remains unresolved.
Use `--tool parallel` for the [earlier parallel-gripper experiment](docs/panda-card-mlx.md).

Activate the existing `simenv` environment, then:

```powershell
python run_policy.py --backend cuda --seconds 20 --seed 42 --output-dir runs/my_episode
python tools/run_scenarios.py --seeds 0 1 --output-dir runs/my_seed_sweep
python validate_sim.py
```

Each episode records policy inputs/actions, bounded commands, evaluator telemetry and video. Physics pauses during inference. Card ground truth is restricted to simulation mechanics, initialization, explicitly scripted demonstrations and evaluation.

## Documentation

- [Documentation index](docs/README.md), [setup](docs/setup.md), [architecture](docs/architecture.md), and [results](docs/results.md).
- [Post-training the policy](docs/post-training.md): data preparation, training approaches, checkpoint integration and evaluation. Training support remains work to implement.
- [Milestone specification](Board_Game_Robot_Milestone_1_Spec.docx).

Documentation is Markdown, readable directly in the repository.

## Directory layout

| Location | Contents |
| --- | --- |
| Root Python scripts | Simulator, scene generation, policy runtime, demonstration collection and validation entry points |
| `tools/` | Setup helpers, policy conversion and experiment diagnostics |
| `assets/` | Runtime scene, camera/controller profiles and suction configuration |
| `docs/` | Markdown guides, historical findings, selected evidence and comparison media |
| `end_effector/` | Separate end-effector design study and its supporting assets |
| `models/`, `vendor/` | Downloaded checkpoints and dependencies; ignored by Git |
| `datasets/`, `runs/` | Local demonstrations and complete experiment outputs; ignored by Git |

Keep new experiments in fresh directories under `runs/` and datasets under `datasets/`. Preserve original observations and logs when summarizing results. The documentation includes a small, attributed evidence bundle; full recordings and checkpoints stay in local or shared artifact storage.
