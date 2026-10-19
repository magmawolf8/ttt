# Initial pipeline and frozen-policy investigation

> Development record from 16 October 2026. Read the [current results](../results.md) for the project status. References marked **local artifact** identify files in the original workspace; they are not included in this site.

Recorded before the RGB intervention. The hybrid experiments are archived in [rgb-centering.md](hybrid-control.md). The later [seed/adaptation audit](seed-audit.md) found a graphics-startup confound, limited generalization beyond the original seed and a CUDA/JAX numerical-tolerance failure on another input; it qualifies the single-input and single-seed findings below. Setup/provenance is in [setup.md](../setup.md); interfaces and physics are in [architecture.md](../architecture.md).

## CPU and GPU pipeline

The initial JAX CPU closed-loop test executed 15 model-generated commands but took 37.0–43.7 seconds per chunk. The CUDA implementation subsequently ran every parameter on `cuda:0`, without CPU offload. Its initial three-cycle run took 1.61 seconds for the first inference and 0.64–0.70 seconds for subsequent calls, with 6.63 GiB peak allocation. Neither early test picked up the card.

A five-warm-call benchmark measured a median 0.547 seconds per chunk, about 70× faster than the CPU warm-call median of 38.36 seconds. On the exact saved CPU observation and reconstructed flow noise, CUDA actions differed from JAX by relative L2=0.003828 (0.38%), maximum absolute difference about 0.00482. This is a single-input conversion check, not universal numerical equivalence. Artifacts: GPU conversion validation (`runs/pi05_cuda/conversion_validation.json`; local artifact), early CPU run (`runs/pi05/results.json`; local artifact), early GPU run (`runs/pi05_cuda/results.json`; local artifact).

## Six full 20-second adapter experiments

All used scene/noise seed 42, 400 commands and frozen weights. These are development ablations of one scene, not independent success-rate trials.

| Configuration / run directory | Closest TCP-to-card distance | Physical contact time | Valid pickup |
| --- | ---: | ---: | --- |
| `pi05_cuda_20s` — original adapter | 232.95 mm | 0 s | No |
| `pi05_cuda_20s_mirror` — image convention only | 106.39 mm | 0 s | No |
| `pi05_cuda_20s_calibrated` — cameras and controller | 10.63 mm | 0 s | No |
| `pi05_cuda_20s_upright` — corrected + upright cup | 0.521 mm | 1.626 s | No |
| `pi05_cuda_20s_refined` — RGB256 + slower approach | 26.18 mm | 0 s | No |
| `pi05_cuda_20s_rgb256` — RGB256 + normal schedule | 0.480 mm | 0.794 s | No |

Directories are under `runs/`. Comparison metrics (`runs/calibration_audit/episode_comparison.json`; local artifact), detailed diagnosis (`runs/calibration_audit/diagnosis.json`; local artifact), 20-second comparison video (`runs/calibration_audit/before_after_20s.mp4`; local artifact). Run `python tools/summarize_episodes.py` to regenerate them.

The original episode contained 80 calls and lasted 160.53 wall seconds excluding loading. Minimum requested tool height was 215.6 mm: it never attempted to reach the tabletop. Median commanded/actual directional cosine was 0.995. The card appeared in every external observation (75 pixels initially) and was absent from the wrist view in only three late observations. A same-seed scripted demonstrator succeeded with 116.6 mm clearance. This localized the original failure to approach rather than the suction mechanism.

## Verified adapter fixes

**Image convention.** Upstream robosuite returns raw OpenGL pixels and the LIBERO OpenPI example reverses both axes. Native MuJoCo RGB already reverses the vertical axis, so only horizontal mirroring is needed. An audit against raw framebuffer pixels matched both cameras exactly (maximum pixel error zero).

**Camera calibration.** The external camera now uses the reference LIBERO tabletop pose translated to the local table and a 45° vertical FOV. The wrist camera's relative rotation was 180° wrong and its mount too close to the cup. Correcting the reference hand/gripper transforms gave cup-relative position `[0,0.05,-0.095]` m and a 180° Y rotation. The corrected view shows part of the tool, but UR wrist occlusion remains and the appearance differs from a Franka gripper. Other wrist offsets did not resolve this mismatch.

**Limits and tool normal.** The legacy controller limited each action to 5 mm translation, 0.03 rad rotation and 0.03 rad joint-target change. Corrected limits are 15 mm, 0.10 rad and 0.12 rad, while preserving workspace, joint and actuator force bounds. A table-normal cup constraint limits tilt to eight degrees without card-pose access. The reference translation/rotation scales remain 0.05 m / 0.5 rad. XYZ and all world rotation direction tests passed. The original manufacturer tool0 axis was already correct.

**Performance.** Caching rendering contexts per resolution removed repeated context creation for every video frame. The corrected upright episode took 62.31 wall seconds for 20 simulated seconds, median inference 0.609 s. The first episode's 160.53 seconds included substantial recording overhead.

**Proprioception.** Fixed-noise input probes showed exactly zero output change when all eight state values were zeroed. The checkpoint uses `discrete_state_input=False` and has no separate continuous-state projection. Therefore the corrected virtual Z offset (0.8→0.9 m) cannot by itself repair this model. Enabling state tokens would change its training-time interface and was rejected. Image orientation and card-position changes do affect outputs; a more explicit blue-card prompt had little effect on that input. These results do not show that the policy ignores vision.

## Remaining edge failure

The corrected upright run first contacted the card at about 3.3 s. Its cup center was 28.37 mm from the card's short-axis center, versus a valid maximum of 22.5 mm for a 9 mm radius cup on a 63 mm card. The cup overhung by **5.87 mm**. Normal error was 6.94°, below the 12° gate. All 33 command-rate contact samples were outside the interior footprint, and no 500 Hz attachment-eligible samples were observed. The simulator correctly refused to weld.

Slower near-table motion (2 mm commands, replanning every two steps) stalled roughly 26 mm above the card; RGB256 resizing alone also failed. Neither became the default. Franka/gripper→UR7e/vacuum transfer is a plausible contributor to the residual aiming error, rather than a proven sole cause. No distant weld, relaxed seal footprint, smaller cup, ground-truth targeting or scripted lift was introduced in these six experiments.

## Independent checks and retained evidence

Three scripted seeds passed pickup/hold/release. Empty-space vacuum, actual edge contact and force overload passed negative tests. NaN policy actions were rejected without motion. Original quantile normalizers had finite values and positive ranges. These checks establish functioning mechanics and adapters, not learned competence.

- Adapter audit (`runs/calibration_audit/adapter_audit.json`; local artifact): `python tools/audit_adapter.py`.
- Fixed-noise input probes (`runs/pi05_cuda_20s/input_probes.json`; local artifact): `python tools/probe_policy_failure.py`.
- Corrected simulator validation (`runs/simulator_validation_suction_upright/results.json`; local artifact): `python validate_sim.py`.
- Every full episode retains raw observations/actions, `results.json`, `telemetry.jsonl`, `diagnostics.json`, `visibility.json`, `policy_rollout.mp4` and `episode_dashboard.mp4`.

## Primary references

- [Pinned OpenPI LIBERO example](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/libero/main.py): image preprocessing, state/action interface and action schedule.
- [Pinned OpenPI model configuration](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/config.py): exact `pi05_libero` settings.
- [robosuite v1.4.1 Panda hand camera](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/robosuite/models/assets/robots/panda/robot.xml), [gripper frame](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/robosuite/models/assets/grippers/panda_gripper.xml), [OSC configuration](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/robosuite/controllers/config/osc_pose.json).
- [LIBERO tabletop environment](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/envs/problems/libero_tabletop_manipulation.py): reference tabletop/camera conventions.
- [Manufacturer robot description](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description), [MuJoCo Python renderer](https://mujoco.readthedocs.io/en/stable/python.html).
