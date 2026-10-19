"""Check that simulation/camera/controller settings do not depend on the seed."""
import ast
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from card_sim import CardSim
from tools.audit_render_reproducibility import array_hashes


def main():
    out=ROOT/"runs/seed_diagnosis/seed_contract"
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for profile in ["suction_upright","reference_scale_audit"]:
        baseline=None
        for seed in [0,1,42,2,3,4,5]:
            sim=CardSim(seed,profile=profile)
            try:
                values={"model":array_hashes(sim.model),"robot_qpos":sim.data.qpos[:6].copy(),
                        "camera_position":sim.data.cam_xpos.copy(),"camera_orientation":sim.data.cam_xmat.copy(),
                        "profile":dict(sim.policy_profile)}
                if baseline is None:
                    baseline=values
                rows.append({"profile":profile,"seed":seed,
                             "model_arrays_identical":values["model"]==baseline["model"],
                             "robot_qpos_identical":bool(np.array_equal(values["robot_qpos"],baseline["robot_qpos"])),
                             "camera_extrinsics_identical":bool(np.array_equal(values["camera_position"],baseline["camera_position"]) and np.array_equal(values["camera_orientation"],baseline["camera_orientation"])),
                             "controller_profile_identical":values["profile"]==baseline["profile"],
                             "card_xy_evaluation_only_m":sim.data.xpos[sim.card,:2].tolist()})
            finally:
                sim.close()
    conditions={}
    for name in ["card_sim.py","pi05_policy.py","pi05_torch.py"]:
        tree=ast.parse((ROOT/name).read_text())
        conditions[name]=[ast.unparse(node.test) for node in ast.walk(tree) if isinstance(node,ast.If) and any(isinstance(part,ast.Name) and part.id=="seed" for part in ast.walk(node.test))]
    checks={"no_seed_conditioned_physics_or_policy_branches":not any(conditions.values()),
            "all_model_robot_camera_and_profile_invariants_pass":all(all(row[key] for key in ["model_arrays_identical","robot_qpos_identical","camera_extrinsics_identical","controller_profile_identical"]) for row in rows)}
    report={"checks":checks,"seed_conditionals":conditions,"trials":rows,
            "limitation":"This rules out seed-conditioned settings, not overfitting of globally chosen hyperparameters. Uniform adapter ablations and additional seeds are evaluated separately."}
    (out/"results.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
