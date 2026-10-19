# Project notes

- [Setup and reproduction](setup.md): Windows environment, assets, GPU runtime and commands.
- [Simulation and policy contracts](architecture.md): observation/action interfaces, suction physics, data and evaluation boundaries.
- [Initial investigation](prior-investigation.md): CPU/GPU validation, six adapter experiments, camera/control audits and original edge-overhang diagnosis.
- [Unassisted seed runs](unassisted-seeds.md): removal of the auxiliary controllers and current frozen-policy results.
- [Seed regression and adaptation audit](seed-diagnosis.md): graphics-startup fix, controlled scene/noise experiments, adapter ablation and limits of single-seed tuning.
- [Archived RGB centering and scenario results](rgb-centering.md): historical measured results and artifact index; those controllers are removed from the current runtime.

All reported trials are development experiments. They do not constitute the specification's full held-out evaluation. No training or checkpoint edits were performed during either diagnosis. Scripted, learned and hybrid controllers are identified separately.
