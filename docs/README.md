# Documentation

[Read the overview](index.md) or open `index.html` for the styled documentation site.

| Section | Contents |
| --- | --- |
| [Setup](setup.md) | Environment, assets, and commands for current code |
| [Architecture](architecture.md) | Policy/control interfaces, physics, and scoring |
| [Results](results.md) | Current evidence, video, and limitations |
| [Evidence](evidence/index.md) | Included measurements and provenance |
| [Site maintenance](publishing.md) | Build, preview, and future GitHub Pages setup |
| [Seed audit](archive/seed-audit.md) | Detailed rendering, sampling, adapter, and numerical diagnosis |
| [Unassisted runs](archive/unassisted-seeds.md) | Auxiliary controller removal and original two-seed batch |
| [Initial investigation](archive/initial-investigation.md) | Earlier CPU/GPU and adapter experiments |
| [Hybrid experiments](archive/hybrid-control.md) | Archived RGB centering and scripted completion |

Edit Markdown, then run `python tools/build_docs.py`. Commit Markdown and generated HTML together. `docs/site.json` controls page order and navigation; `docs/static/site.css` controls presentation. The HTML is committed so GitHub Pages can serve `main` → `/docs` directly. See [site maintenance](publishing.md) for the complete workflow.
