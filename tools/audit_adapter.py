"""Audit pixels, Cartesian command frames, limits, and mechanics without weights."""
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from card_sim import CardSim
from validate_sim import pickup


def main():
    report={"image_convention":{},"cartesian_signs":[],"rotation_signs":[]}
    sim=CardSim(42,profile="suction_upright")
    try:
        for camera in ["external","wrist"]:
            upright=sim.render(camera)
            raw=np.empty_like(upright)
            mujoco.mjr_readPixels(raw,None,mujoco.MjrRect(0,0,224,224),sim.renderer._mjr_context)
            reference=raw[::-1,::-1]  # Upstream LIBERO OpenGL -> OpenPI conversion.
            actual=upright[:,::-1]
            report["image_convention"][camera]={"max_pixel_error":int(np.abs(reference.astype(int)-actual.astype(int)).max()),"pass":bool(np.array_equal(reference,actual))}
        for axis in range(3):
            sim.reset(42)
            before=sim.position
            action=np.zeros(7); action[-1]=-1; action[axis]=.05
            for _ in range(5):
                sim.apply_policy_action(action)
            delta=sim.position-before
            report["cartesian_signs"].append({"axis":axis,"motion_m":delta.tolist(),"pass":bool(delta[axis]>.001)})
        for axis in range(3):
            sim.reset(42)
            before=sim.rotation
            action=np.zeros(7); action[-1]=-1; action[axis+3]=.02
            for _ in range(5):
                sim.apply_policy_action(action)
            delta=Rotation.from_matrix(sim.rotation@before.T).as_rotvec()
            report["rotation_signs"].append({"axis":axis,"rotation_world_rad":delta.tolist(),"pass":bool(delta[axis]>.005)})
        sim.reset(42)
        report["same_seed_pickup"]=pickup(sim)
        sim.set_suction(False)
        sim.step_physics(250)
        report["release_pass"]=not bool(sim.data.eq_active[sim.eq]) and bool(sim.data.xpos[sim.card,2]<.001)
        q=sim.data.qpos.copy()
        rejected=False
        try:
            sim.apply_policy_action(np.full(7,np.nan))
        except ValueError:
            rejected=True
        report["nonfinite_rejected_without_motion"]=bool(rejected and np.array_equal(q,sim.data.qpos))
        report["source_translation_scales_m"]=[.05]*3
        report["source_rotation_scales_rad"]=[.5]*3
        report["profile"]=sim.policy_profile
        report["all_passed"]=all(item["pass"] for item in report["image_convention"].values()) and all(item["pass"] for item in report["cartesian_signs"]+report["rotation_signs"]) and report["same_seed_pickup"]["held_2s"] and report["release_pass"] and report["nonfinite_rejected_without_motion"]
    finally:
        sim.close()
    destination=ROOT/"runs/calibration_audit/adapter_audit.json"
    destination.parent.mkdir(exist_ok=True,parents=True)
    destination.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__=="__main__":
    main()
