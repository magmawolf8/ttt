"""Small demonstration pipeline with episode splits and train-only statistics.

NPZ recordings are model-independent; state/action keys match the OpenPI LIBERO adapter.
An adaptation experiment must recollect sufficient data and train its own normalizers.
"""
import argparse
import json

import numpy as np

from card_sim import CardSim, ROOT


def collect(seed,profile="suction_upright"):
    sim=CardSim(seed,profile=profile)
    card=sim.data.xpos[sim.card].copy()  # Expert only.
    records={key:[] for key in ["image","wrist_image","state","actions","attached","suction","qpos","time"]}
    def record_step(action):
        obs=sim.observation()
        records["image"].append(obs["observation/image"])
        records["wrist_image"].append(obs["observation/wrist_image"])
        records["state"].append(obs["observation/state"])
        records["actions"].append(np.array(action,dtype=np.float32))
        records["attached"].append(bool(sim.data.eq_active[sim.eq]))
        records["suction"].append(sim.suction)
        records["qpos"].append(sim.data.qpos[:6].copy())
        records["time"].append(sim.data.time)
        sim.apply_policy_action(action)
    def go(target,steps,suction):
        for _ in range(steps):
            delta=(np.asarray(target)-sim.position)/.05
            record_step([*np.clip(delta,-1,1),0,0,0,1 if suction else -1])
    try:
        go([card[0],card[1],.06],80,False)
        go([card[0],card[1],.00035],80,True)
        go([card[0],card[1],.00035],20,True)
        go([card[0],card[1],.12],80,True)
        go([card[0],card[1],.12],40,True)
        success=bool(sim.data.eq_active[sim.eq]) and bool(sim.data.xpos[sim.card,2]>.05)
        arrays={key:np.asarray(values) for key,values in records.items()}
        arrays["prompt"]=np.array("Pick up the face-down card.")
        metadata={"seed":seed,"success":success,"frames":len(arrays["actions"]),"source":"scripted expert","suction_events":sim.events,"interventions":len(sim.interventions),"config":sim.config,"simulation_profile":profile,"policy_profile":sim.policy_profile}
        return arrays,metadata
    finally:
        sim.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--train-episodes",type=int,default=2)
    parser.add_argument("--validation-episodes",type=int,default=1)
    parser.add_argument("--sim-profile",default="suction_upright")
    args=parser.parse_args()
    out=ROOT/"datasets"/("card_smoke_"+args.sim_profile)
    out.mkdir(parents=True,exist_ok=True)
    manifest={"format":"NPZ arrays, one complete episode per file","control_hz":20,"source":"scripted expert with evaluator-only card pose","held_out_test_seed_start":20000,"test_seeds_collected":False,"episodes":[],"training_ready":False,"note":"Tiny pipeline validation dataset; not a sufficient adaptation dataset."}
    training=[]
    for split,count,base in [("train",args.train_episodes,100),("validation",args.validation_episodes,10100)]:
        for index in range(count):
            arrays,metadata=collect(base+index,args.sim_profile)
            if not metadata["success"]:
                raise RuntimeError(f"Demonstration pickup failed: {metadata}")
            if not all(np.isfinite(arrays[key]).all() for key in ["state","actions","qpos","time"]):
                raise RuntimeError("Invalid demonstration values")
            if arrays["image"].shape[1:]!=(224,224,3) or arrays["actions"].shape[1:]!=(7,):
                raise RuntimeError("Unexpected demonstration schema")
            file=f"{split}_{index:04}.npz"
            np.savez_compressed(out/file,**arrays)
            metadata.update(split=split,file=file)
            manifest["episodes"].append(metadata)
            if split=="train":
                training.append(arrays)
            print(json.dumps({k:v for k,v in metadata.items() if k not in ["config","suction_events"]}),flush=True)
    stats={}
    for key in ["state","actions"]:
        values=np.concatenate([episode[key] for episode in training])
        stats[key]={"mean":values.mean(0).tolist(),"std":values.std(0).tolist(),"q01":np.quantile(values,.01,axis=0).tolist(),"q99":np.quantile(values,.99,axis=0).tolist()}
    (out/"norm_stats.json").write_text(json.dumps({"norm_stats":stats},indent=2))
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print(f"Validated recordings and train-only normalizers: {out}",flush=True)


if __name__=="__main__":
    main()
