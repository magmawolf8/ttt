# Card manipulation with a frozen robot policy

A student robotics project exploring a UR7e arm that can manipulate board-game pieces. The first milestone uses a suction tool to pick up a playing card in MuJoCo, guided by an open-source vision-language-action policy.

> **Current status:** the simulation and GPU policy pipeline work. Reliable card pickup has not been demonstrated. The latest development evidence is from 16 October 2026.

## Working pipeline

Two RGB cameras and a language instruction feed the frozen `pi05_libero` policy. Its commands pass through an experimental bounded Cartesian adapter to the UR7e simulation. Suction attachment is evaluated by contact physics.

| Component | Current implementation |
| --- | --- |
| Robot and tool | UR7e with an 18 mm suction cup |
| Simulator | MuJoCo, 500 Hz physics |
| Policy | Frozen OpenPI pi0.5, RTX 4070 inference |
| Inputs | External and wrist RGB, instruction, compatibility state |
| Control | 20 Hz; replan after five actions |
| Evaluation | Attach and hold the lowest card point ≥50 mm above the table for ≥2 seconds |

The runtime contains no RGB centering helper or scripted lift. Card pose and simulator segmentation are excluded from policy/control inputs. The [architecture](architecture.md) documents the information boundaries and physical approximations.

## What the experiments show

None of 14 controlled, 20-second development episodes passed the full pickup criterion. One scene/noise combination briefly lifted the card about 67 mm, but sustained the required clearance for only 0.222 seconds. Rendering startup, stochastic sampling, scene placement and backend numerics all affect behavior. The settings developed on seed 42 have not established generalization.

Read [current results and the video](results.md) for the measurements, or the [full seed audit](archive/seed-audit.md) for the causal checks and unresolved questions. Historical hybrid successes used an auxiliary controller that has since been removed.

## Start here

1. [Set up and run an episode](setup.md).
2. [Understand the simulation and policy contracts](architecture.md).
3. [Review the results and included evidence](results.md).
4. [Maintain or publish these docs](publishing.md).

The simulator models a rigid card and a conditional suction weld. Flexible cards, peeling, decks, physical deployment, foundation-model orchestration and training remain outside this prototype.
