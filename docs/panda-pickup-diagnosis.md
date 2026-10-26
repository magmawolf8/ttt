# Why the fine-tuned pilot does not pick up the card

The completed demonstration video is scene seed **10100**, using the local
MLX checkpoint selected at training step 150. The immediate failure is lack of
contact: the cup remains **at least 29.6 mm from the card surface** throughout
20 simulated seconds. It cannot form a vacuum seal at that distance.

At its closest approach (11.522 s), vacuum is on, tilt is only 0.509 degrees
and the footprint condition is satisfied. The cup is about 31.6 mm laterally
from the card center, although the rotated card still accommodates the cup at
that instant. There are **zero contact steps, zero eligible seal steps and zero
attachments**. This is not a failure of holding force after attachment.

The camera perspective hides the vertical separation. The previous dashboard
displayed card clearance above the table, which stayed near zero and did not
describe the cup's distance. The dashboard now labels **card lift** and
**cup-to-card gap** separately.

## What was checked

`tools/audit_panda_pickup.py` performs a recorded-action audit, an explicitly
scripted demonstration replay, teacher-forced model evaluation, and a batch
alignment check. Evidence is in [panda-pickup-diagnosis.json](panda-pickup-diagnosis.json).

| Check | Result |
| --- | --- |
| Raw neural actions versus 400 executed commands | Exact match |
| Vacuum state versus sign of action output 6 | Exact match |
| Same-placement recorded expert replay | Successful pickup |
| Replayed RGB/state versus recorded observations | All 239 frames exact for both cameras and state |
| Expert replay's maximum card clearance | 119.3 mm |
| Expert replay's continuous valid hold | 4.162 s |
| Batch-of-two versus independent action-expert velocities | Maximum absolute difference 3.81e-6 |

The replay uses saved scripted labels. It verifies that these observations and
actions are aligned and that the controller, cup and seal rules can complete
this exact placement. It is **not** a learned-policy success and does not add
an assist to policy inference.

## The learned behavior is wrong before contact

After its initial approach, the policy alternates between upward and downward
commands. From 5–10 s its mean vertical action is -0.0057, and from 10–15 s it
is +0.0017: these commands largely cancel. The successful expert continues
descending to contact at approximately 6.10 s, then establishes a seal.

We also fed the model saved observations from the successful expert trajectory,
so this check does not require the policy to recover from its own earlier
position errors. Ten-action chunks were sampled with seed 42; the first five
actions from each anchor were compared with recorded labels, matching the
runner's replanning interval.

During the demonstration's descent phase, the expert's mean vertical command
was **-0.0998**, but the policy predicted **+0.0087** on average and **60% of the
50 tested actions pointed upward**. Vacuum signs agreed with the labels for all
those actions. During the seal phase, all 25 tested predictions pointed upward.
The pilot has not learned the required approach/contact/lift sequence reliably,
even on these successful-path observations.

## What the evidence supports

The immediate cause and incorrect vertical predictions are measured. The
training setup used only 16 successful episodes and selected a checkpoint by
aggregate flow-matching validation loss. The lower loss therefore did not prove
successful physical behavior. More varied, contact-focused and recovery
demonstrations, phase-specific action diagnostics and closed-loop development
evaluation are the next useful changes.

This audit does **not** isolate episode count, frozen visual features, or
ambiguity between descent and lift observations as the sole learning cause.
It also does not justify weakening the seal conditions or adding automatic
downward motion during inference. The seven-value action interface already
controls the cup correctly; no second arm is needed for this failure.

The earlier multi-scene evaluation was interrupted after this first completed
development episode. Its second incomplete recording is preserved and is not
counted as a completed trial. The separate pretrained test baseline completed
all three reserved placements and succeeded on one.

```bash
.venv-mlx/bin/python tools/audit_panda_pickup.py \
  --rollout runs/panda_sft_v1_candidate_eval/episode_000_seed_10100 \
  --demonstration datasets/panda_suction_sft_v1/validation_0000.npz \
  --checkpoint runs/panda_sft_mlx_v1/checkpoint \
  --output-dir runs/my_pickup_audit
```
