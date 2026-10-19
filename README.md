# UR7e card manipulation in MuJoCo

A UR7e with a suction tool, two RGB cameras and frozen OpenPI pi0.5 inference on an RTX 4070 Laptop GPU. The task is `Pick up the face-down card.`

The current runtime uses frozen pi0.5 through an experimental, bounded action adapter. RGB centering and the scripted seal/lift controller have been removed. No training or task-specific weight changes are used. [The seed audit](docs/seed-diagnosis.md) records a rendering-startup fix, strong scene/noise sensitivity and the limits of earlier single-seed tuning; reliable pickup remains unachieved.

## Run

Activate the existing `simenv` conda environment, then:

```powershell
# Frozen policy, corrected camera/control adapter:
python run_policy.py --backend cuda --seconds 20 --seed 42 --output-dir runs/my_policy_episode
# Two unassisted seeds, reusing one model on the GPU:
python tools/run_scenarios.py --seeds 0 1 --output-dir runs/my_seed_sweep
# Independent scripted simulator checks:
python validate_sim.py
```

Each episode records raw policy actions, observations, adapter bounds, evaluator telemetry and a 20-second video. Physics pauses during inference. Ground-truth card pose is restricted to simulation mechanics, scene initialization, separately labelled scripted demonstrations and evaluation.

Read [the documentation index](docs/README.md) for setup, interfaces, results, videos and limitations. The simulator uses a rigid card and conditional suction weld; flexible cards, peeling and decks remain unmodeled. Physical deployment and foundation-model orchestration are outside this prototype.
