# Original suction cup on Panda, unchanged pretrained pi0.5

The Panda/card runner now defaults to the original 18 mm suction cup. The
`pi05_libero` weights, tokenizer, action normalizer and MLX model code remain
unchanged. No training is performed. The parallel gripper remains selectable
with `--tool parallel`; the separate native LIBERO runner is unchanged.

The same scene and suction adapter are also available through
`run_panda_card.py --backend cuda`. See [NVIDIA setup](panda-cuda.md); the pickup
results below were measured on MLX, not CUDA.

```bash
# Original suction cup, three recorded trials:
.venv-mlx/bin/python run_panda_card_mlx.py --tool suction --seeds 42 43 44

# Live viewer:
.venv-mlx/bin/mjpython run_panda_card_mlx.py --tool suction --viewer --seeds 44 --policy-noise-seed 42

# Explicitly scripted adapter/mechanics validation (not a learned-policy run):
.venv-mlx/bin/python tools/check_panda_suction.py --output-dir runs/my_cup_validation
```

Use the existing [Panda/MLX environment](libero-mlx.md). No additional packages
or model downloads are needed.

## Action-head contract

The checkpoint has a padded internal action dimension of 32 and returns ten
seven-value actions after the existing LIBERO output transform. That interface
is unchanged:

| Output | Interpretation |
| --- | --- |
| 0–2 | Native Panda OSC_POSE translation deltas, scaled once by 0.05 m |
| 3–5 | Native Panda OSC_POSE rotation deltas, scaled once by 0.5 rad |
| 6 > 0 | Close command → vacuum ON |
| 6 ≤ 0 | Open/zero command → vacuum OFF, matching the original UR7e suction mapping |

The first six values pass straight to the native controller. The seventh is
one logical vacuum channel, not a physical finger joint. There are no extra
learned output heads or dummy actuators. The observation interface retains eight
state values, with two virtual finger indicators encoding the commanded vacuum
state, as in the original UR7e setup. They do not expose attachment or card pose;
this particular checkpoint ignores state input anyway.

The cup face is aligned with the original Panda grip site, 97 mm along the
gripper mounting axis. A mounting collar positions the original cup assembly at
this offset. Its control-frame orientation also matches the original Panda
gripper. Thus motion commands target the cup face at the expected tool point.
The original cup's stem, contact geometry, mass, inertia and contact parameters
are copied from `assets/scene.xml`. Panda's wrist camera is retained.

This makes the interface compatible without retraining. It does not establish
that a policy trained with parallel fingers will infer successful suction
approaches, seal timing or lift behavior. Images and physical grasp behavior
still differ from the training setup.

## Original physical suction rules

`panda_suction.py` binds the existing `CardSim` suction implementation to the
Panda's MuJoCo model/data. It reuses the exact seal, release and holding methods,
rather than introducing easier attachment rules. The card/table physics and
external camera are preserved from the prior Panda/card experiment.

The vacuum command alone cannot attach the card. The cup must contact it, fit
fully inside its footprint, be within 12 degrees of its normal, move slower than
0.04 m/s relative to it, and maintain eligibility for 40 ms. The original
conditional weld models the seal, with a 2 N holding-force limit and an 8 mm
constraint-error limit. Vacuum OFF releases it. Ground-truth geometry is used
only inside this physical mechanism, reset and evaluation, never by the policy
or arm controller. No automatic centering, orientation correction or scripted
lift is enabled during inference.

Success retains the original suction criterion: the card is attached and its
lowest point stays at least 50 mm above the table for two continuous seconds.

## Verification

The reproducible adapter/mechanics checks in `tools/check_panda_suction.py`
passed all 74 checks in `runs/panda_suction_mechanics_02`. These cover preserved
scene/tool physics, the seven-value action interface, TCP position/orientation,
matching initial arm joints, command signs, 5 mm motion requested by an x action
of 0.1, observation shape, no-contact rejection, aligned attachment, sustained
lift, release, overload and edge rejection. The **scripted** fixture lifted the
card to 119.3 mm clearance and held the valid lift for 5.15 seconds.

The first scripted fixture stopped slightly short of contact with the compliant
OSC controller. Its failure is preserved in `runs/panda_suction_mechanics_01`.
The corrected fixture requests gentle contact pressure; this changes only the
test trajectory, not the policy controller or physical suction thresholds.

Frozen-policy results are recorded separately in `runs/panda_suction_pretrained_01`
and summarized in [panda-suction-validation.json](panda-suction-validation.json).
Each run includes raw neural actions, executed commands, vacuum/attachment state,
sampled noise, observations, generated scene XML, source snapshots and videos.

## Pretrained result — 25 October 2026

**1/3 placements succeeded** with unchanged weights, using scene seeds 42, 43,
44 and a fixed sampling seed of 42. Every trial lasted 20 simulated seconds.
All 1,200 executed actions exactly matched saved neural outputs, all vacuum
commands matched the sign of output 6, and no saved policy observation contained
card pose or attachment feedback. Checkpoint-manifest and MLX-model hashes match
the preceding parallel-gripper experiment.

| Scene seed | Result | Maximum card clearance | Valid lift held continuously |
| --- | --- | --- | --- |
| 42 | Contact without a seal | 0.051 mm | 0 s |
| 43 | Approach/alignment failure | 0.051 mm | 0 s |
| 44 | Successful suction pickup | 288.7 mm | 2.296 s |

Seed 44 attached and released several times as the model changed its gripper
command, then maintained a seal and lifted the card. Its final clearance was
261.9 mm. This is a learned-policy success using the original physical suction
model, separate from the scripted mechanism test. One success in three scenes
does not establish reliable pickup across placements.

The live-viewer repeat (`runs/panda_suction_live_44`) also succeeded. All 80
observation, noise and action chunks were identical to the original seed-44
episode. This is a reproducibility check, not a fourth independent placement.

The seed-42 closest approach had approximately 15.6 degrees of cup tilt and
failed the cup-footprint condition; neither the 12-degree normal limit nor the
footprint rule was relaxed to manufacture a seal.
