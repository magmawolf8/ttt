# UR7e card manipulation in MuJoCo

A prototype toward a robot arm that manipulates board-game pieces: a UR7e suction tool, two RGB cameras, and frozen OpenPI pi0.5 inference on an RTX 4070 Laptop GPU. The current instruction is `Pick up the face-down card.`

The minimum simulation-to-policy pipeline runs. Reliable pickup remains unachieved: none of the 14 controlled development episodes met the full hold criterion. The adapter is experimental; RGB centering and scripted lifting have been removed. See [current results](docs/results.md) and the [detailed seed audit](docs/archive/seed-audit.md).

## Run

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
