# Documentation

The project explores a UR7e arm that can manipulate board-game pieces. The current prototype uses suction to pick up a rigid playing card in MuJoCo, guided by a frozen pi0.5 policy. GPU inference works; reliable pickup has not been demonstrated. Start with the setup and current results, then read the post-training guide for the next development steps.

| Section | Contents |
| --- | --- |
| [Setup](setup.md) | Environment, assets, and commands for current code |
| [Architecture](architecture.md) | Policy/control interfaces, physics, and scoring |
| [Post-training](post-training.md) | Demonstrations, training methods, checkpoint integration and evaluation |
| [Results](results.md) | Current evidence, video, and limitations |
| [Evidence](evidence/index.md) | Included measurements and provenance |
| [Seed audit](archive/seed-audit.md) | Detailed rendering, sampling, adapter, and numerical diagnosis |
| [Unassisted runs](archive/unassisted-seeds.md) | Auxiliary controller removal and original two-seed batch |
| [Initial investigation](archive/initial-investigation.md) | Earlier CPU/GPU and adapter experiments |
| [Hybrid experiments](archive/hybrid-control.md) | Archived RGB centering and scripted completion |

Edit these Markdown files directly. Add guides to this index and keep links relative. Historical investigations stay in `archive/`; current findings belong in the main guides. References marked **local artifact** identify files in the original workspace, outside the compact evidence bundle.

Keep original report/media bytes intact so their hashes continue to match `evidence/manifest.json`. Large recordings, datasets and checkpoints belong in the ignored workspace directories or shared storage, with paths and hashes recorded in the relevant guide.
