# Unassisted pi0.5 seed runs

> Development record from 16 October 2026. Read the [current results](../results.md) for the project status. References marked **local artifact** identify files in the original workspace; they are not included in this site.

On 2026-10-16, the RGB centering and scripted seal/lift controller were removed at the user's request. The current execution path is frozen pi0.5 → bounded action adapter → MuJoCo. Historical hybrid results/source snapshots remain archived under `runs/` and [the earlier notes](hybrid-control.md).

## Removed and retained behavior

- Removed `rgb_servo.py`, its audit and replay tools, all three RGB-assistance CLI options, the idealized seal-switch accessor, and the per-action auxiliary correction/completion path.
- `run_policy.py` now passes each neural action directly to `CardSim.apply_policy_action`. It performs no card detection, image-derived centering, vacuum latch, prescribed lift or attachment-conditioned action change.
- Kept the calibrated external/wrist cameras, original checkpoint/normalizers, 20 Hz command rate, five-action replanning, 500 Hz physics and the `suction_upright` adapter. Its translation/rotation/workspace/joint bounds and eight-degree table-normal tilt constraint remain.
- Card pose/attachment state remain internal to suction physics and offline scoring. Independent scripted simulator validation/demonstration tools are separately labelled and are not called by these policy episodes.

No training or weight changes were made. The scenario initialization state-preservation repair remains in place. `tools/run_scenarios.py` now supports multiple seeds with a directly injected, shared GPU policy, without loader monkeypatching or auxiliary controllers.

## Run protocol

```powershell
conda activate simenv
python tools/run_scenarios.py --scenarios baseline --seeds 0 1 --output-dir runs/pi05_unassisted_seeds_0_1
```

Each seed controls a fresh scene for the full 20 simulated seconds: 400 commands and 80 policy calls. The seed initializes both card XY/yaw and Torch sampling noise. The GPU model is loaded once and reused serially; Torch sampling is reseeded for each episode. Physics pauses during inference, and videos follow simulated time.

Both runs record `controller_source: frozen_pi05`, per-run source snapshots/SHA-256 hashes, raw policy actions, observations, bounded-command telemetry, physics scoring and videos. No auxiliary source file is included in the new snapshots.

Success still requires valid suction attachment with the lowest card point at least 50 mm above the table for two continuous seconds. Results are development examples, not a success-rate estimate.

## Results

Both episodes completed in full on the RTX 4070, with every model parameter on `cuda:0`.

| Seed | Closest TCP-to-card surface distance | Cup/card contact | Attachments | Successful pickup | Wall time excluding loading |
| --- | ---: | ---: | ---: | --- | ---: |
| 0 | 211.49 mm | 0 s | 0 | No | 84.53 s |
| 1 | 233.42 mm | 0 s | 0 | No | 85.39 s |

Warm median inference latency was 0.828 s/chunk for seed 0 and 0.891 s/chunk for seed 1. Each run executed 400 commands, 80 policy calls and 10,000 physics steps. Both failed during approach, before card contact or seal formation; neither exercised the edge-overhang boundary seen in the earlier seed-42 run. The policy moved the robot and changed vacuum commands, but did not bring the cup to these card placements. These two examples do not establish an overall success rate.

The recorded initial card XY positions, read only for offline evaluation, were (0.43237,-0.09181) m and (0.51406,-0.07117) m respectively.

## Artifacts and verification

- Machine-readable batch results (`runs/pi05_unassisted_seeds_0_1/summary.json`; local artifact).
- Seed 0 dashboard video (`runs/pi05_unassisted_seeds_0_1/seed_0/baseline/episode_dashboard.mp4`; local artifact), results (`runs/pi05_unassisted_seeds_0_1/seed_0/baseline/results.json`; local artifact), diagnostics (`runs/pi05_unassisted_seeds_0_1/seed_0/baseline/diagnostics.json`; local artifact).
- Seed 1 dashboard video (`runs/pi05_unassisted_seeds_0_1/seed_1/baseline/episode_dashboard.mp4`; local artifact), results (`runs/pi05_unassisted_seeds_0_1/seed_1/baseline/results.json`; local artifact), diagnostics (`runs/pi05_unassisted_seeds_0_1/seed_1/baseline/diagnostics.json`; local artifact).
- 20-second comparison video (`runs/pi05_unassisted_seeds_0_1/seed_comparison_20s.mp4`; local artifact).
- Direct-action and video verification (`runs/pi05_unassisted_seeds_0_1/verification.json`; local artifact).

For all 800 commands, the adapter's recorded raw input exactly matches its saved neural action. Vacuum state matches the policy action's sign on every command, with no latch. No auxiliary-controller telemetry is present. Both dashboard videos contain exactly 400 frames at 20 fps. Removed source files are absent, and the modified runtime files compile.

Follow-up: [the seed diagnosis](seed-audit.md) found a graphics-startup regression: the shared-model runner imported Torch before the first OpenGL context and selected a different rendering adapter from standalone runs. It is fixed, and actual GL adapter metadata is now recorded. The original results above remain historical observations; use the follow-up's rendering-consistent reruns and crossed experiments for comparisons. Seed 42 still contacts the card, while seeds 0/1 still miss. The follow-up also audits whether globally chosen adapter settings overfit seed 42.
