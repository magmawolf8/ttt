"""Scripted simulator checks; these are never reported as learned policy results."""
import json
from pathlib import Path
import time
import argparse

import imageio.v2 as imageio
import numpy as np

from card_sim import CardSim, DOWN, ROOT


def move(sim, target, duration, suction=False, frames=None):
    start = sim.position
    for fraction in np.linspace(0,1,max(2,round(duration*20))):
        sim.command_pose(start+(np.asarray(target)-start)*fraction,DOWN,suction)
        if frames is not None:
            frames.append(sim.render(width=640,height=480))


def pickup(sim, frames=None):
    card = sim.data.xpos[sim.card].copy()  # Scripted demonstrator ground truth only.
    move(sim,[card[0],card[1],.06],2,frames=frames)
    move(sim,[card[0],card[1],.00035],2,suction=True,frames=frames)
    for _ in range(20):
        sim.command_pose([card[0],card[1],.00035],DOWN,True)
        if frames is not None:
            frames.append(sim.render(width=640,height=480))
    attached = bool(sim.data.eq_active[sim.eq])
    move(sim,[card[0],card[1],.12],2,suction=True,frames=frames)
    minimum = 1.
    for _ in range(40):
        sim.command_pose([card[0],card[1],.12],DOWN,True)
        rot = sim.data.xmat[sim.card].reshape(3,3)
        extent = np.abs(rot[2]) @ np.array([.0315,.044,.00015])
        minimum = min(minimum,float(sim.data.xpos[sim.card,2]-extent))
        if frames is not None:
            frames.append(sim.render(width=640,height=480))
    held = bool(sim.data.eq_active[sim.eq]) and minimum >= .05
    return {"attached":attached,"held_2s":held,"minimum_clearance_m":minimum,"events":sim.events.copy()}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--sim-profile",default="suction_upright")
    parser.add_argument("--output-dir",type=Path,default=None)
    args=parser.parse_args()
    out = args.output_dir or ROOT/"runs"/("simulator_validation_"+args.sim_profile)
    out.mkdir(parents=True,exist_ok=True)
    sim = CardSim(profile=args.sim_profile)
    results = {}
    sim.set_suction(True)
    sim.step_physics(100)
    results["no_contact_does_not_attach"] = not bool(sim.data.eq_active[sim.eq])
    trials=[]
    for seed in range(3):
        sim.reset(seed)
        frames=[] if seed==0 else None
        result=pickup(sim,frames)
        result["seed"]=seed
        trials.append(result)
        if frames:
            imageio.mimsave(out/"scripted_pickup.mp4",frames,fps=20,macro_block_size=1)
        if seed==0:
            from PIL import Image
            Image.fromarray(sim.render(width=960,height=640)).save(out/"lift.png")
        sim.set_suction(False)
        sim.step_physics(250)
        results[f"release_seed_{seed}"] = bool(not sim.data.eq_active[sim.eq] and sim.data.xpos[sim.card,2]<.001)
    results["aligned_contact_attaches"]=all(t["attached"] for t in trials)
    results["gentle_lift_holds"]=all(t["held_2s"] for t in trials)
    # Reuse a valid pickup, then exceed the finite holding force.
    sim.reset(0)
    pickup(sim)
    sim.data.xfrc_applied[sim.card,2]=-5.
    sim.step_physics(10)
    results["overload_releases"]=not bool(sim.data.eq_active[sim.eq])
    sim.data.xfrc_applied[:]=0
    # An actual edge contact is rejected by the cup footprint rule.
    sim.reset(0)
    card=sim.data.xpos[sim.card].copy()
    card_rot=sim.data.xmat[sim.card].reshape(3,3)
    edge=card+card_rot@np.array([.029,0,0])
    move(sim,[edge[0],edge[1],.06],2)
    move(sim,[edge[0],edge[1],.00035],2,True)
    sim.step_physics(100)
    results["edge_contact_does_not_attach"]=not bool(sim.data.eq_active[sim.eq])
    report={"controller":"scripted with card ground truth", "simulation_profile":args.sim_profile,"checks":results,"trials":trials,"full_acceptance_evaluation":False}
    (out/"results.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    sim.close()
    if not all(results.values()):
        raise SystemExit(1)


if __name__=="__main__":
    main()
