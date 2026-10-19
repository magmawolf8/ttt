# UR7e card manipulation in MuJoCo

A UR7e suction tool, two RGB cameras, and frozen OpenPI pi0.5 inference on an RTX 4070 Laptop GPU. The instruction is `Pick up the face-down card.`

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

- [Project overview](docs/index.md), [setup](docs/setup.md), [architecture](docs/architecture.md), and [results](docs/results.md).
- [Documentation site instructions](docs/publishing.md): preview locally, rebuild, and enable GitHub Pages later.
- [Milestone specification](Board_Game_Robot_Milestone_1_Spec.docx).

Open `docs/index.html` directly, or run `python -m http.server 8000 --bind 127.0.0.1 --directory docs` and visit `http://127.0.0.1:8000`.

Model weights, downloaded dependencies, datasets and complete run outputs are ignored by Git. The documentation includes a small, attributed evidence bundle and one comparison video. No remote repository or public site has been configured.
