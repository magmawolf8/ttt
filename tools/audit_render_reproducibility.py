"""Compare identical simulated resets across model/GL-context lifetimes."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from OpenGL import GL

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from card_sim import CardSim


def array_hashes(obj):
    result={}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        value=getattr(obj,key)
        if isinstance(value,np.ndarray) and value.dtype.kind in "biufc":
            result[key]=hashlib.sha256(value.tobytes()).hexdigest()
    return result


def differences(a,b):
    return {key:{"changed_values":int(np.count_nonzero(a[key]!=b[key])),
                 "mean_absolute_error":float(np.abs(a[key].astype(float)-b[key].astype(float)).mean()),
                 "max_absolute_error":float(np.abs(a[key].astype(float)-b[key].astype(float)).max())}
            for key in ["observation/image","observation/wrist_image","observation/state"]}


def main():
    out=ROOT/"runs/seed_diagnosis/render_audit"
    out.mkdir(parents=True,exist_ok=True)
    baseline=None
    models=None
    states=None
    rows=[]
    for index in range(5):
        if index==3:
            import torch
            torch.cuda.init()
        sim=CardSim(42)
        try:
            obs=sim.observation()
            repeated=sim.observation()
            model_hashes=array_hashes(sim.model)
            state_hashes=array_hashes(sim.data)
            if baseline is None:
                baseline=obs
                models=model_hashes
                states=state_hashes
            row={"iteration":index,"cuda_initialized":index>=3,
                 "gl_vendor":GL.glGetString(GL.GL_VENDOR).decode(),
                 "gl_renderer":GL.glGetString(GL.GL_RENDERER).decode(),
                 "vs_first":differences(baseline,obs),"repeat_same_context":differences(obs,repeated),
                 "different_model_arrays":[k for k in models if models[k]!=model_hashes.get(k)],
                 "different_data_arrays":[k for k in states if states[k]!=state_hashes.get(k)]}
            rows.append(row)
            np.savez_compressed(out/f"observation_{index}.npz",**obs)
        finally:
            sim.close()
    (out/"results.json").write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows,indent=2))


if __name__=="__main__":
    main()
