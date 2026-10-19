# RGB centering, pickup and scene variations

**Archived experiment.** The user subsequently requested removal of RGB centering and scripted lifting. `rgb_servo.py`, its audit/replay tools, the seal-switch accessor and all auxiliary CLI modes have been removed from the active runtime. The results and source snapshots below are retained as historical evidence, not current behavior. See [unassisted seed runs](unassisted-seeds.md) for the current pipeline. All commands and default choices below describe the historical implementation and no longer apply to the current runtime.

Development experiments on 2026-10-16. No training, task-specific checkpoint edits, smaller cup or relaxed suction criteria were used. No controller reads simulator card pose, depth or segmentation IDs. The successful result below is **hybrid control**, not a claim that frozen pi0.5 learned suction pickup.

## What fixed the edge error

`rgb_servo.py` receives a native 640 × 480 external RGB frame, explicit camera calibration, robot TCP position/orientation and the raw pi0.5 action. It has no MuJoCo imports or simulation object. A color mask selects the blue card back; connected components are checked against approximate card size and the workspace. Pixel rays intersect the calibrated Z=0 table plane to estimate XY. These are image-derived estimates, not simulator poses.

The estimate is filtered over five observations, then frozen before final approach to avoid chasing the exposed portion of a card occluded by the tool. Once TCP height falls below 120 mm, lateral motion is corrected toward that estimate, at at most 4 mm per command. Downward motion is paused while estimated centering error exceeds 5 mm. The cup is aligned to the calibrated horizontal table normal. Pi0.5 still supplies approach before this guard, descent when centered, and vacuum/lift commands in `--rgb-centering` mode.

The first implementation disabled the guard again above 120 mm. Policy upward motion repeatedly escaped it, producing a failed approach. Latching the guard once entered fixed this. An initial reset audit also found a false ambiguity in one of 14 seeds: small border fragments passed the component shape test. Scaling the minimum component area with image resolution removed those fragments. The final audit found all 14 estimates within 1.1 mm of evaluation truth (strict check <5 mm). Ground truth is read only after detector output, for this offline measurement.

## Centering alone versus completion

| Full 20-second experiment, seed 42 | Valid attachments | Maximum clearance | ≥50 mm for ≥2 s |
| --- | ---: | ---: | --- |
| Corrected frozen policy, no RGB assistance | 0 | approximately 0 mm | No |
| Initial height-gated RGB guard | 0 | approximately 0 mm | No |
| Latched RGB centering, policy lift/vacuum | 15 | 14.30 mm | No |
| RGB centering + seal-switch completion | 1, no releases | 119.69 mm | **Yes, 11.53 s** |

Artifacts:

- [Centering-only result](../runs/pi05_rgb_centering_latched_seed42/results.json), [diagnostics](../runs/pi05_rgb_centering_latched_seed42/diagnostics.json), [video](../runs/pi05_rgb_centering_latched_seed42/episode_dashboard.mp4).
- [Successful hybrid result](../runs/pi05_rgb_pickup_seed42/results.json), [diagnostics](../runs/pi05_rgb_pickup_seed42/diagnostics.json), [20-second video](../runs/pi05_rgb_pickup_seed42/episode_dashboard.mp4).
- [Initial failed guard](../runs/pi05_rgb_centering_seed42/results.json), [RGB audit](../runs/rgb_servo_audit/results.json), [unchanged-mechanics regression checks](../runs/simulator_validation_rgb_revision/results.json).

The centering-only run established that the original edge-overhang barrier was removable using camera pixels. At command rate, its final estimate was about 0.43 mm from the original card center, and its final TCP-to-estimate XY error about 0.40 mm. It accumulated 13.302 s of contact/eligible physics samples and 11.256 s attached. However, the policy repeatedly opened vacuum and did not sustain a lift. Gripper closure produces no corresponding finger motion in this suction embodiment; the policy also ignores proprioception, so it lacks that potential grasp-state cue. This is a plausible explanation for repeated grasp attempts, not an isolated causal proof.

In the successful hybrid's first attached command sample, cup-center offsets in the card's frame were only 0.055 mm and -0.710 mm, safely inside the unchanged ±22.5/35 mm footprint limits. These are evaluator measurements after control, not targets supplied to the servo.

`--rgb-pickup` explicitly adds a nonlearned completion controller. After the idealized digital vacuum seal switch activates, it latches vacuum, settles for five 20 Hz commands, then raises the TCP at no more than 40 mm/s to Z=120 mm and holds. Seal loss enters a stationary hold rather than continuing to lift. The switch is simulated from the mechanism's seal state; it supplies no object pose. Real hardware would need a pressure sensor, appropriate thresholds and debounce. This is an added sensor assumption, clearly recorded in run metadata.

The successful hybrid run recorded 400 commands, 80 CUDA policy calls and exactly 400 video frames at 20 fps. All model parameters were on `cuda:0`; peak allocation was 6.63 GiB. Wall time was 87.70 s excluding model loading. The model continues to be evaluated and its raw actions logged during assisted lift/hold, but the completion controller overrides those commands. That phase is not evidence of learned lifting.

## Information and reproducibility audit

`CardSim` and the evaluator have access to card state for physics/scoring. The RGB servo is a separate module with only numeric calibration, RGB, TCP state, actions and (in completion mode) a boolean seal switch. Scenario card poses are set only before episodes. Evaluation card poses and segmentation visibility never flow into the helper.

Raw neural actions remain in `actions_*.npy`; `rgb_servo.json` and each telemetry record retain raw and modified actions, image detection, estimated target, state and intervention mode. `rgb_servo_used.py` preserves the helper version. Later scenario runs also snapshot relevant source files and record SHA-256 hashes in `results.json`. No weight files were inspected during this diagnosis beyond normal loading for inference.

The initial failed guard's actual first wrist input differed from the earlier baseline in six color-channel values, despite identical reset state and saved initial images. Its first action chunk differed by 1.21% relative L2 and its unassisted trajectory diverged. The latched and successful hybrid experiments had byte-identical first policy inputs **and identical first action chunks** to the upright baseline. This matters when interpreting the first failed experiment: it was not a perfectly matched trajectory comparison. No GPU numerical defect was established.

## Scene variations

`scene_scenarios.py` provides independent initialization-only variants:

| Scenario | Change |
| --- | --- |
| `rotated` | Card XY=(0.49,-0.075) m, yaw=75° |
| `left` | Card XY=(0.41,-0.08) m, yaw=-30° |
| `dim` | Overhead diffuse ×0.35, headlight ambient/diffuse ×0.5 |
| `heavy_slippery` | Card 9 g, sliding friction coefficient 0.25 on card/table |
| `red_card` | Untextured red back, intentionally outside blue detector support |
| `missing` | Card outside the controller workspace |

The reset detector audit passed all seven scenario expectations: <5 mm error on the five supported setups, rejection for red/missing targets. Rejection holds robot translation/rotation and disables vacuum. The complete episode suite preserves individual videos and reports under `runs/rgb_scenarios`; [suite summary](../runs/rgb_scenarios/summary.json).

The initial complete suite found successful pickup under dim lighting (119.69 mm maximum clearance, 10.03 s valid hold). Both changed-placement cases failed during approach. Final telemetry corrects an initial interpretation from sparse progress logs: they **did** briefly enter the 120 mm region (minimum TCP heights 117.26 mm rotated / 115.82 mm left), activating the latched XY guard for 217 / 191 commands respectively, but subsequently rose again without contact. Remaining vertical approach/descent therefore matters even after XY guidance. Red and missing targets produced zero joint displacement, zero suction events and vacuum off throughout all 400 commands, which is the intended rejection behavior. These negative cases are not pickup successes.

### Mass-variant setup bug and repair

The original `heavy_slippery` run is **invalid as a mass/friction comparison**. Calling `mj_setConst` to recompute mass-dependent constants overwrote live `MjData` using the model's default pose. This moved the card to (0.45,0) and changed the robot's initial configuration, confounding the intended variation. The artifact is retained, marked invalid in the aggregate metrics, and excluded from pickup claims.

The corrected implementation saves/restores the complete `mjSTATE_INTEGRATION` state around constant recomputation. A direct check after the scenario's settling period measured exactly zero TCP displacement, zero card XY displacement and zero arm joint-position error relative to the original reset, while card mass was correctly 0.009 kg. [State preservation evidence](../runs/rgb_servo_audit/state_preservation.json). This repair preserves the seeded pose and changes only the intended physical properties.

### Follow-up controller tests

Two small follow-up revisions were evaluated in full episodes under `runs/rgb_followup`:

- `--rgb-center-from-start` applies the same image-derived lateral target throughout approach, testing whether earlier XY guidance improves approach. It still leaves vertical approach/descent and pre-seal vacuum to the policy; it is an additional explicit assistance mode.
- The completion controller anchors XY to the robot's own TCP position when the seal switch activates. The first successful implementation issued zero lateral increments during lift, which accumulated 7.92 mm lateral TCP excursion while lifting. The anchor corrects this using proprioception, with no card-pose input.

These changes do not silently replace the original centering-only comparison or successful hybrid recording. Source snapshots and separate run directories preserve both versions.

| Scenario | Final-approach RGB + seal/lift | RGB from start + anchored seal/lift |
| --- | --- | --- |
| Seed-42 baseline | **Pickup**, 119.69 mm, 11.534 s valid hold | No pickup; policy hovered |
| Left placement / -30° yaw | No pickup; no contact | **Pickup**, 119.71 mm, 6.110 s valid hold |
| Rotated placement / 75° yaw | No pickup; no contact | No pickup; policy hovered |
| Dim lighting | **Pickup**, 119.69 mm, 10.032 s valid hold | Not run |
| Corrected 9 g / low friction | Not run as a live-policy episode; see replay below | No pickup; policy hovered |
| Red back / missing target | **Rejection checks pass**, stationary with vacuum off | Not run |

All listed live episodes completed 20 simulated seconds with 400 commands and 80 GPU policy calls. The left-placement success's maximum attached TCP XY excursion was 0.245 mm. [Follow-up summary](../runs/rgb_followup/summary.json), [left-placement GPU video](../runs/rgb_followup/left/episode_dashboard.mp4), [all development metrics](../runs/rgb_experiments/summary.json).

**Default choice:** keep final-approach centering when `--rgb-pickup` is selected. Early centering helps the left placement but makes the baseline hover, so it remains an explicit experimental flag rather than a universal improvement. The XY anchor and state-preservation repair are retained. Across these examples the successful frozen-policy pipeline requires explicit RGB and seal/lift assistance; pure learned pickup remains unachieved. These outcomes do not establish a success rate.

### Controlled replay checks

`tools/replay_servo_regression.py` replays the original successful episode's 400 raw policy actions through current RGB/seal control and physics. It performs **zero live policy inference calls**. This isolates the XY-anchor change and corrected mass setup from neural trajectory changes; it is a controller/mechanics regression, not a new live closed-loop policy success.

| Replay scenario | Maximum clearance | Valid hold above 50 mm | Maximum attached TCP XY excursion |
| --- | ---: | ---: | ---: |
| Baseline, anchored completion | 119.69 mm | 11.530 s | 0.218 mm |
| Corrected 9 g / friction 0.25 | 119.67 mm | 11.514 s | 0.231 mm |

Both formed one seal without release. The original unanchored baseline excursion was 7.924 mm. [Replay reports and explicit controller labels](../runs/rgb_replay_regression/summary.json), [baseline replay video](../runs/rgb_replay_regression/baseline/replay_regression.mp4), [heavy-card replay video](../runs/rgb_replay_regression/heavy_slippery/replay_regression.mp4). The heavier-card result demonstrates adequate simulated holding under these gentle trajectories, without implying a live-policy success on that variant.

## Reproduce

```powershell
python run_policy.py --backend cuda --seconds 20 --seed 42 --rgb-centering --output-dir runs/my_centering_trial
python run_policy.py --backend cuda --seconds 20 --seed 42 --rgb-pickup --output-dir runs/my_pickup_trial
python tools/run_scenarios.py --output-dir runs/my_scenarios
python tools/run_scenarios.py --scenarios baseline heavy_slippery rotated left --center-from-start --output-dir runs/my_followup
python tools/audit_rgb_servo.py
python tools/replay_servo_regression.py
python validate_sim.py --output-dir runs/my_mechanics_checks
python tools/summarize_rgb_experiments.py
```

The suite reuses a single frozen GPU model serially; concurrent full models do not fit 8 GB VRAM. `--scenario NAME` also works directly with `run_policy.py`. The comparison tool creates [machine-readable metrics](../runs/rgb_experiments/summary.json) and a [20-second comparison video](../runs/rgb_experiments/edge_fix_comparison.mp4).

Final verification: all 13 new live GPU episode videos and both replay videos contain exactly 400 frames at 20 fps. Modified Python files compile, local documentation links resolve, all RGB reset/scenario checks pass, and all positive/negative mechanics regression checks pass. [Verification record](../runs/rgb_experiments/verification.json).

## Limits

The RGB helper assumes one stationary blue card on a known horizontal plane and calibrated camera; it is not an arbitrary-card detector. Occlusion, similarly colored distractors, moving cards, stacked cards and camera-calibration error remain open problems. A held card cannot be localized in 3D by intersecting its pixels with the table plane; freezing the XY estimate before approach avoids using that invalid projection during lift.

The suction model remains a rigid-card conditional weld with finite holding limits. It does not simulate pneumatic pressure, seal compliance, flex, peeling or deck separation. Scenario trials are exploratory examples, not a held-out success rate. Pure pi0.5 still needs embodiment/task adaptation or a more capable policy; this laptop experiment demonstrates a frozen-policy pipeline and an explicit no-card-pose hybrid workaround.
