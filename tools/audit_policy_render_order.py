"""Isolate rendering/model initialization order without inspecting policy weights."""
import json
from pathlib import Path
import sys

import numpy as np
from OpenGL import GL

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from card_sim import CardSim
from pi05_policy import load_policy
from tools.audit_render_reproducibility import array_hashes,differences


def main():
    policy_first="--policy-first" in sys.argv
    out=ROOT/"runs/seed_diagnosis"/("render_order_policy_first" if policy_first else "render_order")
    out.mkdir(parents=True,exist_ok=True)
    policy=load_policy(backend="cuda") if policy_first else None
    first=CardSim(42)
    original=first.observation()
    model_hashes=array_hashes(first.model)
    state_hashes=array_hashes(first.data)
    rows=[]
    with np.load(ROOT/"runs/pi05_seed42_regression_check/observation_000.npz") as saved:
        reference={key:saved[key] for key in saved.files}
    def capture(label,sim):
        obs=sim.observation()
        np.savez_compressed(out/(label+".npz"),**obs)
        models=array_hashes(sim.model);states=array_hashes(sim.data)
        rows.append({"label":label,"gl_renderer":GL.glGetString(GL.GL_RENDERER).decode(),
                     "policy_loaded_before_first_gl_context":policy_first,
                     "vs_saved_standalone":differences(reference,obs),
                     "vs_before_policy_load":differences(original,obs),
                     "different_model_arrays":[k for k in model_hashes if model_hashes[k]!=models.get(k)],
                     "different_data_arrays":[k for k in state_hashes if state_hashes[k]!=states.get(k)]})
        return obs
    try:
        capture("before_load",first)
        if policy is None:
            policy=load_policy(backend="cuda")
        capture("same_context_after_load",first)
        for index in range(3):
            sim=CardSim(42)
            try:
                obs=capture(f"new_context_{index}_after_load",sim)
                policy.infer(obs)
                capture(f"new_context_{index}_after_infer",sim)
            finally:
                sim.close()
        capture("original_context_after_inference",first)
    finally:
        first.close()
    (out/"results.json").write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows,indent=2),flush=True)


if __name__=="__main__":
    main()
