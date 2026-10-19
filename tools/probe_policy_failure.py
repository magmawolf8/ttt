"""Fixed-noise probes to separate input/transfer failures from suction mechanics."""
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from card_sim import CardSim
from pi05_policy import load_policy


def object_pixels(sim,camera):
    sim.render(camera)
    sim.renderer.enable_segmentation_rendering()
    try:
        seg=sim.renderer.render()
        geom=seg[...,1]==int(mujoco.mjtObj.mjOBJ_GEOM)
        return {"card":int(((seg[...,0]==sim.card_geom)&geom).sum()),"cup":int(((seg[...,0]==sim.cup_geom)&geom).sum())}
    finally:
        sim.renderer.disable_segmentation_rendering()


def main():
    out=ROOT/"runs/pi05_cuda_20s"
    recorded=np.load(out/"observation_000.npz")
    obs={key:recorded[key] for key in recorded.files}
    obs["prompt"]=obs["prompt"].item()
    policy=load_policy(backend="cuda")
    noise=np.random.default_rng(2026).standard_normal((10,32)).astype(np.float32)
    sim=CardSim(42,profile="legacy")
    report={"noise_seed":2026,"single_observation_probes_only":True,"probes":[],"initial_visibility":{cam:object_pixels(sim,cam) for cam in ["external","wrist"]}}
    base=None
    def probe(name,inputs):
        nonlocal base
        actions=policy.infer(inputs,noise=noise)["actions"]
        if base is None:
            base=actions.copy()
        mean=actions[:5,:3].mean(0)
        result={"name":name,"mean_first_five_translation_outputs":mean.tolist(),"mean_close_command":float(actions[:5,6].mean()),"max_action_difference_from_baseline":float(np.abs(actions-base).max()),"first_actions":actions[:5].tolist()}
        report["probes"].append(result)
        print(json.dumps({k:v for k,v in result.items() if k!="first_actions"}),flush=True)
    try:
        probe("baseline",obs)
        probe("zero_proprioception",{**obs,"observation/state":np.zeros(8,dtype=np.float32)})
        probe("horizontally_mirrored_images",{**obs,**{key:np.ascontiguousarray(obs[key][:,::-1]) for key in ["observation/image","observation/wrist_image"]}})
        probe("images_rotated_180",{**obs,**{key:np.ascontiguousarray(obs[key][::-1,::-1]) for key in ["observation/image","observation/wrist_image"]}})
        probe("explicit_blue_card_prompt",{**obs,"prompt":"Pick up the blue playing card from the table."})
        positions=[]
        for x in [.38,.52]:
            sim.reset(42)
            sim.data.qpos[sim.card_qpos]=x
            sim.data.qpos[sim.card_qpos+1]=0.
            sim.data.qvel[:]=0
            mujoco.mj_forward(sim.model,sim.data)
            sample=sim.observation()
            probe(f"card_x_{x:.2f}",sample)
            positions.append({"x_m":x,"card_offset_from_tcp_x_m":float(x-sim.position[0]),"external_visibility":object_pixels(sim,"external"),"wrist_visibility":object_pixels(sim,"wrist")})
        report["target_position_probes"]=positions
        # Check that the exact same seed/card is mechanically reachable by the
        # separately labelled scripted demonstrator, independent of the VLA.
        from validate_sim import pickup
        sim.reset(42)
        report["same_seed_scripted_control"]=pickup(sim)
        report["sources"]={"checkpoint_config":"https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/config.py","libero_input_preprocessing":"https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/libero/main.py"}
    finally:
        sim.close()
        (out/"input_probes.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({"initial_visibility":report["initial_visibility"],"same_seed_scripted_control":report["same_seed_scripted_control"]},indent=2),flush=True)


if __name__=="__main__":
    main()
