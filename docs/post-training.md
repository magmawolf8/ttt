# Post-training the manipulation policy

This guide describes the work needed to adapt pi0.5 to UR7e suction manipulation. The repository currently provides **card-only simulation, frozen-policy inference, diagnostics and a small scripted demonstration recorder**. It does not yet provide a validated training dataset, task-specific trainer or trained-policy deployment path. Read [setup](setup.md), the [simulation contracts](architecture.md) and [current results](results.md) first. Reliable learned pickup remains unachieved.

## Choose the starting model and training environment

Use the shared Linux GPU server or cloud allocation for training; keep the laptop for simulation, inspection and compatible inference. Pin the OpenPI revision, dependencies, CUDA stack and model configuration in a separate training environment. The Windows environment snapshot is not a portable training lock. Measure memory with the actual batch, image sizes, action horizon and optimizer before launching a long run.

OpenPI presents `pi05_base` as a fine-tuning starting point and `pi05_libero` as a LIBERO-adapted inference checkpoint. Its documented workflow converts data, defines training transforms/configuration, then serves the resulting checkpoint. These are useful upstream building blocks, not completed integration in this repository. [OpenPI training guide](https://github.com/Physical-Intelligence/openpi#fine-tuning-base-models-on-your-own-data).

Starting from `pi05_base` is appropriate for developing an embodiment-specific configuration. Continuing from the existing `pi05_libero` checkpoint provides continuity with recorded results, but also inherits its conventions. Neither choice is proven better here. Keep the choice explicit and compare complete configurations, including state conditioning and normalization. The local LIBERO setup disables state input; supplying better proprioception alone therefore cannot improve it. Enabling state conditioning requires a compatible configuration and training.

## Build a trustworthy dataset

`collect_demos.py` records RGB, state, commanded actions and metadata, but its default output is only a smoke dataset. Its success flag uses final attachment and card-center height; the official evaluator requires the lowest card point to remain at least 50 mm above the table while attached for two seconds. Re-score recordings with the official criterion and inspect approach, contact, lift and hold behavior before treating them as successful demonstrations.

Record the observation immediately before its associated action. Preserve complete episodes, timestamps, task text, object identity, simulator/profile revision and expert provenance. Store evaluator-only pose, segmentation and attachment diagnostics separately from model features. A scripted expert may use object pose to label training actions; that does not authorize passing pose to the deployed policy.

Split by complete episode and scene configuration before augmentation or statistics. Keep training, development and final evaluation placements separate; nearby frames from one trajectory must not cross splits. Cover varied positions, yaw, successful seal formation and recovery states. More nearly identical trajectories do not establish broader coverage. Keep failed trajectories labelled; blindly cloning failed actions teaches their mistakes.

Convert recordings into the selected trainer's dataset format with an explicit schema. OpenPI's examples use LeRobot. Validate camera names, channel order, image range, prompts, action shape and episode boundaries, then inspect a sampled training batch. Define action-chunk padding and masking at episode ends so targets never continue into another reset. [OpenPI data conversion example](https://github.com/Physical-Intelligence/openpi/tree/main/examples/libero).

## Preserve the action and preprocessing contract

The current model outputs seven values: relative XYZ, relative world rotation and gripper command. `CardSim.apply_policy_action()` clips these to [-1,1], scales translation by 0.05 metres and rotation by 0.5 radians, then applies physical bounds; positive gripper output requests suction. Recorded demonstration actions are already relative commands in this representation. Do not apply another absolute-to-delta transform or multiply by the physical scales twice. Replay exported action labels through the existing adapter to verify displacement, rotation convention and suction sign.

Use the same camera mapping and preprocessing during collection, training and inference. The current corrected profiles mirror native MuJoCo RGB horizontally and produce 224 × 224 policy inputs. A second mirror, different crop or changed camera pose creates a different task. Introduce image augmentation deliberately, checking that targets remain visible and that geometric transforms preserve the intended action semantics.

Compute new normalization statistics from training episodes only, after the chosen data transforms. Package the exact normalizer with each checkpoint and apply its inverse once at inference. Inspect constant dimensions and quantile ranges, particularly rotation labels that may all be zero in scripted data. Reusing upstream statistics is a separate choice requiring compatible action definitions; a related robot name is insufficient. [OpenPI normalization guidance](https://github.com/Physical-Intelligence/openpi/blob/main/docs/norm_stats.md).

## Start with supervised adaptation

A practical first experiment is supervised fine-tuning on successful demonstrations using the model's action-training objective. First overfit a tiny inspected subset to expose broken labels or transforms, then train on varied episodes and evaluate closed-loop behavior. Falling training loss is necessary diagnostic evidence, not proof of successful manipulation. Watch for a policy that approaches correctly but reproduces poor suction timing or cannot recover from its own deviations.

Possible parameter-update choices include full fine-tuning, freezing selected modules, or low-rank adaptation. LoRA keeps base weights fixed and learns smaller update matrices; it reduces trainable state but does not remove the base model or activation memory requirement. Rank and target modules are experimental choices, not established solutions for this scene. [LoRA paper](https://arxiv.org/abs/2106.09685).

Verify support in the selected backend. As checked on 25 October 2026, OpenPI's README lists LoRA training as unsupported in its PyTorch implementation. A JAX LoRA configuration or a separately validated implementation is needed for that route; this repository's CUDA inference loader is not a trainer. [OpenPI backend limitations](https://github.com/Physical-Intelligence/openpi#pytorch-support).

Freezing the visual/language backbone can simplify adaptation, but may restrict learning of unfamiliar visual details; updating more modules increases optimization and retention concerns. Change one major factor at a time and record which parameters actually receive gradients.

## Add corrective data or reinforcement learning later

Once a supervised policy moves sensibly, collect expert corrections at states it actually visits: near-edge approaches, missed contacts and premature lifting. Mix corrections with earlier demonstrations and track their provenance. This addresses the mismatch between expert trajectories and the policy's own state distribution, the motivation behind DAgger-style data aggregation. It requires a reliable expert and intervention procedure, neither of which is a finished integration here. [DAgger paper](https://arxiv.org/abs/1011.0686).

Reinforcement learning is a later engineering option requiring a resettable training interface, reward definition, rollout collection and an update method compatible with this policy. Simulator state may supply a training reward while evaluation observations remain restricted. Audit reward shortcuts: proximity or height alone can reward hovering, launching the object or exploiting the conditional weld. Keep physical pickup success as the external evaluation criterion. Establish a useful supervised initialization before investing in sparse-reward exploration.

## Make trained checkpoints load correctly

Fix these gaps before interpreting a rollout as a fine-tuned policy:

- `run_policy.py` has no checkpoint/configuration CLI selection and reports the original frozen model identity.
- `pi05_policy.load_policy(checkpoint=...)` changes the statistics path, but its CUDA branch calls `load_cuda_model()` without a checkpoint argument. That loader always reads `models/pi05_libero_torch`. Changing the outer argument alone can silently pair new statistics with old weights.
- The configuration and normalizer asset path are LIBERO-specific. Export and load model configuration, weights, transforms and normalizer as one identified bundle.
- Remote inference does not accept the runner's `--policy-noise-seed`. Add a documented server-side RNG reset/seed contract and returned checkpoint metadata before comparing seeded remote runs.

Record source revision, dirty patch, dataset/configuration/checkpoint hashes, training seed and inference backend. The existing CUDA/JAX comparison has a known failed tolerance case; conversion is not guaranteed numerically equivalent. Check identical observations and sampled noise through the actual training/export/inference path, then evaluate using one consistent backend and renderer.

## Work from smoke test to measured behavior

1. Reproduce simulator positive/negative checks and a fresh frozen baseline on the intended evaluation machine.
2. Audit a small dataset, convert it, inspect batches and replay action labels.
3. Run one optimizer step, verify intended gradients, save/reload, and reproduce fixed-input outputs.
4. Overfit a tiny subset, then train on varied data and select checkpoints using development rollouts.
5. Evaluate unseen placements with scene and sampling seeds recorded separately. Report successful episodes, denominators, failure categories, interventions and representative failures alongside videos; repeat training seeds where feasible.
6. Extend beyond cards only after generalizing object geometry, suction eligibility and lowest-point scoring. Current card variations do not constitute multi-object support.

Keep a runnable command and artifact manifest for every completed stage. The immediate tasks are dataset validation, trainer configuration and checkpoint round-trip support; there is no ready project-specific training command yet.
