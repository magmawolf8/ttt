"""Scene initialization only. Never imported by the RGB controller."""
import mujoco
import numpy as np

SCENARIOS=("baseline","rotated","left","dim","heavy_slippery","red_card","missing")


def configure_scenario(sim, name):
    settings={"name":name,"modifications":{},"initialization_only":True}
    if name in ("rotated","left","missing"):
        x,y,yaw={"rotated":(.49,-.075,75.),"left":(.41,-.08,-30.),
                 "missing":(.85,.35,0.)}[name]
        sim.data.qpos[sim.card_qpos:sim.card_qpos+7]=[x,y,.00025,np.cos(np.deg2rad(yaw)/2),0,0,np.sin(np.deg2rad(yaw)/2)]
        sim.data.qvel[6:]=0
        settings["modifications"].update(card_reset_xy_m=[x,y],card_reset_yaw_degrees=yaw)
    elif name=="dim":
        sim.model.light_diffuse[:]*=.35
        sim.model.vis.headlight.ambient[:]*=.5
        sim.model.vis.headlight.diffuse[:]*=.5
        settings["modifications"].update(overhead_diffuse_scale=.35,headlight_scale=.5)
    elif name=="heavy_slippery":
        sim.model.body_mass[sim.card]*=5
        sim.model.body_inertia[sim.card]*=5
        sim.model.geom_friction[sim.card_geom,0]=.25
        sim.model.geom_friction[sim.model.geom("table").id,0]=.25
        # mj_setConst uses qpos0 and overwrites the supplied MjData. Preserve the
        # full live integration state so changing mass does not reset the scene.
        spec=mujoco.mjtState.mjSTATE_INTEGRATION
        state=np.empty(mujoco.mj_stateSize(sim.model,spec))
        mujoco.mj_getState(sim.model,sim.data,state,spec)
        mujoco.mj_setConst(sim.model,sim.data)
        mujoco.mj_setState(sim.model,sim.data,state,spec)
        settings["modifications"].update(card_mass_kg=.009,sliding_friction=.25,live_state_preserved=True)
    elif name=="red_card":
        material=sim.model.geom_matid[sim.card_geom]
        sim.model.mat_texid[material,:]=-1
        sim.model.mat_rgba[material]=[.8,.08,.08,1]
        settings["modifications"].update(card_appearance="untextured red; deliberately unsupported by blue detector")
    elif name!="baseline":
        raise ValueError(name)
    mujoco.mj_forward(sim.model,sim.data)
    sim.step_physics(100)
    return settings
