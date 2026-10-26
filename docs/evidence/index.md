# Included evidence

This small bundle accompanies the development findings from **16 October 2026**. Files are copied byte-for-byte from the original local runs. No experiments were rerun during the documentation cleanup.

## Reports and media

| Included file | Original workspace path | Bytes |
| --- | --- | ---: |
| [controlled-baseline.json](controlled-baseline.json) | `runs/seed_diagnosis/consistent_renderer/summary.json` | 1,957 |
| [adapter-audit.json](adapter-audit.json) | `runs/seed_diagnosis/adapter_audit/results.json` | 10,053 |
| [adapter-audit-plan.json](adapter-audit-plan.json) | `runs/seed_diagnosis/adapter_audit_plan.json` | 2,098 |
| [episode-verification.json](episode-verification.json) | `runs/seed_diagnosis/controlled_run_verification.json` | 9,948 |
| [seed-contract.json](seed-contract.json) | `runs/seed_diagnosis/seed_contract/results.json` | 5,274 |
| [policy-parity.json](policy-parity.json) | `runs/seed_diagnosis/policy_parity/results.json` | 818 |
| [attention-numerics.json](attention-numerics.json) | `runs/seed_diagnosis/attention_numerics/results.json` | 4,305 |
| [pixel-sensitivity.json](pixel-sensitivity.json) | `runs/seed_diagnosis/pixel_sensitivity.json` | 315 |
| [graphics-first.json](graphics-first.json) | `runs/seed_diagnosis/render_order/results.json` | 5,921 |
| [policy-first.json](policy-first.json) | `runs/seed_diagnosis/render_order_policy_first/results.json` | 11,123 |
| [video-verification.json](video-verification.json) | `runs/seed_diagnosis/comparison_video_verification.json` | 100 |
| [seed-comparison.mp4](../media/seed-comparison.mp4) | `runs/seed_diagnosis/controlled_seed_comparison_20s.mp4` | 1,136,504 |
| [seed-comparison.png](../media/seed-comparison.png) | `runs/seed_diagnosis/controlled_seed_comparison_lift.png` | 383,806 |
| [render-context.png](../media/render-context.png) | `runs/seed_diagnosis/render_context_comparison.png` | 99,833 |

## Provenance

The [manifest](manifest.json) records every source path, byte count and SHA-256 digest. Preserve the original bytes when copying or checking these files; line-ending conversion changes their hashes.

The complete workspace retains additional videos, per-step telemetry, observation/action arrays and source snapshots. Those are outside this compact evidence bundle. The [archive](../archive/seed-audit.md) retains their relative paths as local artifact references.

The dataset is exploratory: all 14 controlled episodes failed the full pickup criterion, and some earlier comparisons were confounded by graphics startup or scene initialization. See [current results](../results.md) before interpreting the raw measurements.
