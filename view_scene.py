"""Interactive MuJoCo scene viewer (close its window to stop)."""
import time
import mujoco.viewer
from card_sim import CardSim

if __name__ == "__main__":
    sim=CardSim()
    try:
        with mujoco.viewer.launch_passive(sim.model,sim.data) as viewer:
            viewer.cam.lookat[:]=[.35,0,.25]
            viewer.cam.distance=1.5
            viewer.cam.azimuth=135
            viewer.cam.elevation=-30
            while viewer.is_running():
                start=time.monotonic()
                sim.step_physics(5)
                viewer.sync()
                time.sleep(max(0,.01-(time.monotonic()-start)))
    finally:
        sim.close()
