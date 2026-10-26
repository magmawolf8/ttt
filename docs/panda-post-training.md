# Local GPU post-training for the Panda vacuum tool

`train_panda_mlx.py` performs supervised fine-tuning on the local Apple GPU.
This is a custom MLX implementation of the pinned OpenPI flow-matching objective,
not an officially supported OpenPI trainer. OpenPI documents JAX and PyTorch
training on NVIDIA/Linux. Its workflow is demonstration preparation, consistent
data transforms, training, checkpoint loading, and closed-loop evaluation.
[Official training guide](https://github.com/Physical-Intelligence/openpi#fine-tuning-base-models-on-your-own-data).

The starting checkpoint is our already validated `pi05_libero` MLX export.
We continue training its **430,098,464 action-expert parameters**, including the
action projections, time embedding and adaptive norms. The **2,923,335,408**
vision/language parameters stay frozen. This is ordinary supervised fine-tuning,
not LoRA, reinforcement learning or a replacement action head. The first
experiment keeps the existing state-input setting (disabled), two policy RGB
cameras, ten-action horizon and seven-value output interface.

## Data and control boundary

`collect_panda_demos.py` uses a clearly labeled scripted expert to produce
training demonstrations. It may read the card's simulator pose to produce action
labels. The dataset's policy features contain only the same images, eight-value
state and prompt as inference. Object poses, seal feedback and success telemetry
are saved in separate evaluator files and are never loaded by the trainer.

The observation is recorded immediately before its action. Labels are the native
OSC_POSE deltas, including the original gripper-sign-to-vacuum mapping. No second
delta transform or extra physical scaling is applied. The cup, controller and
seal thresholds are unchanged. Successful demonstrations must meet the original
criterion: attachment with the lowest card point above 50 mm for two continuous
seconds. Failed examples stay on disk and are excluded from successful-action
cloning.

Training and validation are split by entire episode and disjoint scene seeds.
Ten-action chunks never cross an episode boundary; chunks near the end repeat
the episode's final holding action. The trainer uses observation anchors every
five commands, matching the inference replanning interval. Raw recordings still
retain all 20 Hz observations and actions.

The first dataset is a **pipeline pilot**, with 16 successful training episodes
(seeds 100–115) and three successful validation episodes (10100–10102). The
fourth validation attempt, seed 10103, failed and is preserved/excluded. Card
position, yaw and trajectory timing vary, but robot start pose, appearance and
the overall approach strategy remain narrow. This dataset cannot establish
broad robustness. A larger experiment should vary these dimensions, add recovery
examples and reserve a larger untouched evaluation set; simply duplicating the
same trajectory is not useful diversity.

## Objective and normalization

The implementation follows pinned OpenPI
[`Pi0.compute_loss`](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py):

- Normalize the seven action labels, then zero-pad to 32 internal dimensions.
- Sample Gaussian noise and `t = Beta(1.5, 1) * 0.999 + 0.001`.
- Form `x_t = t * noise + (1 - t) * actions`.
- Predict the velocity `noise - actions`, minimizing mean squared error over
  the ten timesteps and all 32 dimensions.
- Use AdamW with beta values 0.9/0.95, epsilon 1e-8, weight decay 1e-10 and
  global gradient-norm clipping at 1.0. The learning rate warms up over 20 steps.

The frozen vision/language prefix is cached without changing its equations.
All model arithmetic remains float32, with MLX TF32 disabled as in inference.
Autodiff uses MLX's documented
[`value_and_grad`](https://ml-explore.github.io/mlx/build/html/usage/function_transforms.html#automatic-differentiation)
over the action-parameter dictionary.

We deliberately retain the original LIBERO normalizer because Panda action
coordinates and scales are unchanged. This avoids silently remapping existing
model outputs when continuing training. It is an experimental choice, not a
claim that old statistics always work best. Upstream recommends comparing reused
statistics with statistics computed from training data only.
[Normalization guidance](https://github.com/Physical-Intelligence/openpi/blob/main/docs/norm_stats.md).

## Run locally

Use the [prepared Panda MLX environment](libero-mlx.md):

```bash
.venv-mlx/bin/python collect_panda_demos.py --output-dir datasets/my_suction_data --train-episodes 16 --validation-episodes 4

.venv-mlx/bin/python train_panda_mlx.py --dataset datasets/my_suction_data --output-dir runs/my_suction_training --steps 400 --batch-size 2 --validation-every 50

.venv-mlx/bin/mjpython run_panda_card.py --backend mlx --checkpoint runs/my_suction_training/checkpoint --tool suction --viewer --seeds 20000 --policy-noise-seed 42
```

The collector returns a nonzero status if any demonstration fails, but preserves
the completed manifest and all recordings. Inspect failures before proceeding;
the trainer filters them by the independent success label. Outputs must use new
directories. Original pretrained weights are never overwritten.

`training.json` identifies the dataset, model/source hashes, trainable tensors,
loss history, gradient norms, selected step and peak memory. A fixed validation
set with fixed noise/times selects the saved checkpoint by flow loss. Selection
does not use final test outcomes. The exported bundle contains all weights,
the exact tokenizer/normalizer and training provenance; reloading verifies
trained tensors exactly. The runner accepts `--checkpoint`, records the actual
bundle hash and labels its video/report as fine-tuned.

The new checkpoint is an MLX bundle. CUDA inference needs a matching export;
passing an MLX checkpoint to `--backend cuda` is rejected explicitly rather than
silently loading the original CUDA weights.

## Correctness checks

The 20-update smoke run completed on the Apple M5 Max, with finite gradients,
19 changed parameter tensors and exact checkpoint tensor round-trip. Validation
flow loss on its four fixed validation anchors fell from 0.06361 to 0.01214.
These are training-pipeline checks, not pickup-success measurements.

`tools/check_mlx_training.py` compares identical actions, images, noise and time
against the pinned upstream JAX FP32 training forward. The measured velocity
relative L2 difference was **5.39e-7**, maximum absolute difference **1.91e-6**,
with losses **0.01877909 (MLX)** and **0.01877911 (JAX)**. An action-input projection
gradient checked through the entire expert against a central finite difference
had **0.139% relative error**. This validates one fixed-input objective/gradient
case; it is not an exhaustive training equivalence guarantee.

```bash
.venv-mlx/bin/python tools/check_mlx_training.py --episode datasets/my_suction_data/train_0000.npz --output runs/my_training_parity.json
.venv-mlx/bin/python -m unittest discover -s tests -v
```

The JAX comparison additionally needs `requirements-mlx-reference.txt`. Training
and inference themselves use MLX. Evaluate closed-loop pickup separately on the
same renderer/backend, with untouched scene seeds and a fixed policy noise seed.

## Pilot rollout diagnosis

The 400-update local run completed and selected step 150 by validation flow loss
(0.005835 versus 0.040992 initially). The first completed development rollout
failed: the cup never contacted the card and stopped at least 29.6 mm away.
The same-placement scripted demonstration replay succeeds with bit-identical
observations. Model predictions on its descent observations frequently point
upward. See [the measured pickup diagnosis](panda-pickup-diagnosis.md).
This pilot is not established as an improvement over the pretrained policy.

## Contact and recovery experiment

The `contact-recovery` collector profile adds normal approaches with varied high
starts, near-contact hovers, lateral offsets, tilted cups, and physical release
followed by reattachment. The intentional setup motions are excluded from the
training labels; recorded actions are expert corrections. Card position and yaw,
approach timing, contact point, descent speed and lift height vary. The expert
stops descending once the existing suction mechanism attaches, rather than
waiting a fixed long interval before lifting. All successful episodes still
have to satisfy the original 50 mm / two-second hold criterion.

The expert uses simulator state only to generate labels. Camera images and the
original policy interface remain the complete inference inputs. No geometry,
seal threshold, force limit or learned rollout action is changed by this mode.

```bash
.venv-mlx/bin/python collect_panda_demos.py --profile contact-recovery --output-dir datasets/panda_suction_recovery_v2 --train-episodes 128 --validation-episodes 16 --seed-start 1000
.venv-mlx/bin/python train_panda_mlx.py --dataset datasets/panda_suction_recovery_v2 --output-dir runs/panda_sft_mlx_v2 --steps 1200 --batch-size 4 --cache-size 1024 --phase-balanced --validation-samples 48 --validation-every 100 --rollout-every 400 --development-seeds 31000 31001 31002
```

`--phase-balanced` samples equally across the available approach, recovery,
descent, contact, lift and hold groups, then uniformly within each group. It
changes sampling, not the flow-matching objective. Episode phases are training
metadata and are never policy inputs. Failed demonstration episodes are excluded.

With `--development-seeds`, checkpoints are selected first by successful learned
pickups in those scenes, with fixed validation flow loss breaking ties. These
development scenes are therefore **not final test data**. Compact action-expert
weights are saved before each development evaluation so interrupted candidates
remain recoverable. Final evaluation uses separate seeds 41000–41005, 20 seconds
per scene and policy-noise seed 42 for both pretrained and retrained policies.
The recorded plan is `runs/panda_sft_v2_plan.json`; results must be reported from
completed simulator trials, separately from expert success and training loss.

Collection completed with **128/128 successful training episodes (21,441
frames)** and **16/16 successful validation episodes (2,657 frames)**. Training
contains 26 normal, 26 hover, 26 lateral, 25 tilted and 25 reattachment-mode
episodes. Of the reattachment-mode setups, 24 training and two validation cases
actually attached before release; the remaining one in each split recovered
from an unsuccessful initial seal attempt. The setup attachment flag preserves
this distinction. Training cup starts span 24–262 mm in height and up to 8.48
degrees of tilt. There are 627 contact and 1,086 lateral-correction frames in
the training split. All successful training holds exceed 3.83 seconds.
