# Panda and parallel gripper in the original card scene

This experiment changes the robot to a Franka Panda with its native parallel-jaw
gripper, while retaining the original card/table task and unchanged pretrained
`pi05_libero` MLX weights. It follows the successful native LIBERO bowl trials
described in [LIBERO on MLX](libero-mlx.md).

## Run

After the [Panda/LIBERO setup](libero-mlx.md#setup-and-macos-compatibility):

```bash
.venv-mlx/bin/python run_panda_card_mlx.py --tool parallel --seeds 42 43 44 --output-dir runs/my_panda_card_test

# Optional live viewer, one scene:
.venv-mlx/bin/mjpython run_panda_card_mlx.py --tool parallel --viewer --seeds 42
```

The default duration is 20 simulated seconds per trial, with 20 Hz commands and
replanning after five commands. `--policy-noise-seed` defaults to 42 and is reset
to that same value for each placement, separating scene variation from sampling
variation. Output directories must be new.

The runner now defaults to the [original suction cup](panda-suction-mlx.md).
Specify `--tool parallel` to reproduce the parallel-gripper experiment on this page.

## What is held fixed

`panda_card_sim.py` reads the original `assets/scene.xml` rather than reconstructing
an approximate card/table. It retains the 63 × 88 × 0.3 mm rigid card, 1.8 g mass,
inertia, texture, table/floor geometry, friction, contact solver parameters,
lighting, 500 Hz timestep and integrator. The external camera matches the
corrected `suction_upright` profile. Both policy images remain 224 × 224 RGB with
the same effective orientation. Initial card position and yaw use the original
random-number sequence for each scene seed.

Before inference, `verify_scene()` compares the compiled card/table/floor physical
parameters with the original simulator and checks that no attachment constraint
exists. The Panda starts with its grip site at the original fixed tool position
`[0.45, 0, 0.23]`, pointing down. Reset IK uses this fixed target, never card pose.
Four zero-motion/open-gripper commands provide the original 100 physics steps
of settling before policy control.

## What changes

The UR7e and suction tool are replaced by the native robosuite Panda arm and
PandaGripper meshes, inertias, joints and actuators. The Panda is mounted directly
on the existing table at the origin. Its native wrist camera replaces the
suction-mounted camera. Seven model actions go directly into its native OSC_POSE
controller and physical fingers, replacing the UR7e bounded IK adapter and vacuum
switch. Thus this compares complete robot/tool/controller setups, not just an
isolated end-effector variable.

The prompt remains `Pick up the face-down card.` The tokenizer, normalizer,
checkpoint, MLX model implementation and denoising settings are unchanged.
There is no training, scripted approach, card-centering feedback or artificial
attachment. Model inputs contain only two RGB images, robot state and the prompt.
As in the checkpoint's original configuration, the supplied state is not used by
the model.

## Evaluation and evidence

At every 2 ms physics step, a separate observer measures finger contacts and the
height of the card's lowest point. Success requires contact with both finger
groups while every part of the card stays at least 50 mm above the table for two
continuous seconds. A pushed, briefly flicked or falling card cannot meet this
criterion. Card pose and contact data are logged only for evaluation.

The local experiment is `runs/panda_card_pretrained_01`. Each trial saves a video,
initial/final images, observations, action chunks, sampled noise, exact executed
actions, generated scene XML and physics diagnostics. The parent directory saves
source snapshots, hashes, model metadata and the scene-preservation checks.
The compact results are in [panda-card-validation.json](panda-card-validation.json).

### Result — 25 October 2026

**0/3 successful pickups**, using scene seeds 42, 43 and 44, sampling seed 42,
and 20 simulated seconds per trial. All 1,200 executed commands matched saved
neural outputs; all 240 policy inputs contained only the documented RGB/state/text
keys. The checkpoint manifest and MLX model source hashes match the successful
native LIBERO bowl run. All 39 compiled-scene preservation checks passed.

| Scene seed | Finger contact | Both fingers contacting | Maximum lowest-point clearance | Result |
| --- | --- | --- | --- | --- |
| 42 | Yes | No | 0.055 mm | Contact without a grasp |
| 43 | No | No | 0.051 mm | Approach/alignment failure |
| 44 | Yes | 39 physics steps (78 ms total) | 0.068 mm | Brief contact without lift |

Those sub-0.1 mm clearances are table-contact motion, not meaningful lifts. The
50 mm / two-second criterion was never approached. Two placements showed better
approach than the earlier UR7e seed-42 episode, but replacing the robot/tool did
not solve the card task in these trials.

An independent replay of all saved seed-42 actions reproduced every recorded
end-effector position exactly. At first finger contact (11.55 s), the projected
card width along the closing axis was 90.8 mm, versus a 78.5 mm actual pad gap
(79.3 mm maximum gap observed in the episode). At the end, projected width was
103.5 mm and the pad gap was 73.5 mm. The policy did not align the fingers to
straddle the card; it contacted one side and the table. This is evidence of an
unsuccessful grasp strategy, not proof that every possible grasp is impossible.
The card's short side is 63 mm, so orientation matters. The replay audit is
`runs/panda_card_pretrained_01/episode_000_seed_42/grasp_geometry_audit.json`.

This is a small zero-shot test, not a benchmark success-rate estimate. Additional
sampling seeds or task-specific training could change the outcome; neither is
claimed by this experiment. The card was never thickened, raised or attached by
a constraint to make the result easier.
