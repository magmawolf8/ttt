# Simulation and policy contracts

## Scene and control

`build_scene.py` converts manufacturer UR7e kinematics, masses, inertia tensors, joint limits and collision meshes. The manufacturer explicitly shares UR5e meshes with UR7e. Inertias are rotated from URDF into MuJoCo body coordinates. The manufacturer's flange/tool0 transforms compose to identity relative to wrist 3; this was checked during the adapter audit. This conversion is not a calibration of a physical robot.

The scene has a horizontal table at world Z=0, a rigid 63 × 88 × 0.3 mm card of mass 1.8 g, and an 18 mm diameter suction cup. Initial card XY/yaw are seeded. The initial robot pose is fixed and independent of card position. MuJoCo steps at 500 Hz, using implicitfast and gravity compensation on the six arm joints. Bounded Cartesian commands run at 20 Hz through damped inverse kinematics and position servos.

`assets/policy_profiles.json` defines image convention, calibrated cameras, limits and the optional table-normal tool constraint. The default `suction_upright` allows 15 mm translation and 0.10 rad rotation per command, with 0.12 rad joint-target increments. The workspace is X=[0.25,0.65], Y=[-0.25,0.25], Z=[0.0002,0.60] metres. Joint limits and actuator force limits remain enabled. All command interventions are logged.

## Suction approximation

Attachment requires actual cup/card contact, the entire circular cup footprint inside the card rectangle, normal error below 12°, relative linear speed below 0.04 m/s and 40 ms continuous eligible dwell. A conditional weld then preserves the measured relative pose. Suction off releases it; force above 2 N or positional constraint error above 8 mm breaks it. See `assets/runtime_config.json`.

These geometric tests belong to the simulated mechanism. They do not guide the policy. The card does not snap to the cup. The approximation does not model pressure dynamics, flexible-card deformation, peeling, seal compliance or deck separation. Negative tests cover empty-space suction, edge contact and force overload; positive tests cover pickup, hold and release.

## Frozen pi0.5

`pi05_policy.py` uses the exact pinned `pi05_libero` architecture and its original normalizers: bf16, ten denoising steps, ten predicted actions, normally five executed before replanning. Inputs are external and wrist RGB images (224 × 224 uint8), an instruction and an eight-element float32 robot state. Corrected profiles horizontally mirror native MuJoCo RGB to match upstream LIBERO/OpenPI's raw OpenGL preprocessing.

State consists of TCP XYZ plus `[-0.45,0,0.9]` metres, a world rotation vector in radians, and virtual finger positions `[0.04,-0.04]` with vacuum off or `[0,0]` with vacuum on. **This checkpoint ignores the state values:** its upstream configuration disables state tokens, and its pi0.5 action expert has no separate continuous-state projection. Fixed-noise zero-state probes confirmed identical output. The state is retained for interface compatibility and other controllers.

Seven unnormalized outputs give relative XYZ, relative world rotation and gripper close/open. Position scales by 0.05 m and rotation by 0.5 rad before bounds. Rotations left-multiply current orientation. Positive gripper output requests suction. The upright guard limits tilt to eight degrees relative to the known table normal without using card pose.

GPU loading uses locally converted PyTorch shards. Upstream layout conversion functions were used verbatim, with 811 tensor names/shapes validated at conversion. The two unused language-generation heads were omitted from action inference. Meta construction avoids duplicate model allocation; deterministic rotary/position buffers are reconstructed; selected upstream fp32 normalization/embedding/projection parameters retain their prescribed precision. No training, task-specific weight edits or CPU weight offload is involved. Later diagnosis reads code/configuration and inference outputs, not weight contents.

## Information boundaries and scoring

The learned policy receives RGB/proprioception/instruction, never card pose, depth or simulator segmentation. Its raw actions pass directly to the bounded adapter. RGB target estimation, auxiliary centering, and scripted seal-triggered lifting have been removed. `scene_scenarios.py` sets up scenes before episodes and is outside this controller.

Ground truth remains available only to the suction mechanism, scene initialization, clearly labelled scripted demonstrations and offline evaluation. `EpisodeDiagnostics` evaluates the lowest card point every 2 ms. Success requires a valid attachment and ≥50 mm table clearance continuously for ≥2 seconds within the full requested 20-second episode. Videos/telemetry may display evaluation values; those values are never sent back into control.

No seal/attachment signal is sent to the controller. The former idealized seal-switch accessor and hybrid completion path have been removed. Suction attachment state is used internally by the simulated mechanism and by the evaluator only.

The scripted demonstrations use exact card pose and are never reported as learned results. `collect_demos.py` records synchronized RGB, state, actions, joints, vacuum, attachment, time and instruction. Smoke splits use training seeds 100/101, validation 10100; final-test seeds begin at 20000 and remain uncollected. Statistics use only training episodes. Three successful legacy recordings remain in `datasets/card_smoke`; corrected collections go to profile-specific folders. This small recording check is not a training dataset or evidence of learned competence.

Physical deployment, foundation-model orchestration, training and the specification's full 100-trial evaluation are outside the completed minimum pipeline.
