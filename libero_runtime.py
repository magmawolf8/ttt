"""Local paths and upstream preprocessing for the pinned LIBERO simulator."""
import math
import os
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
LIBERO_REVISION = "f78abd68ee283de9f9be3c8f7e2a9ad60246e95c"


def configure_paths():
    # Avoid LIBERO's interactive first-import prompt and writes to ~/.libero.
    import yaml
    root = ROOT / "vendor/libero/libero/libero"
    if not root.is_dir():
        raise FileNotFoundError("Prepare Panda dependencies with setup_libero_mlx.sh (Mac) or setup_panda_cuda.py (NVIDIA).")
    # The upstream checkout has a namespace outer directory; its editable
    # packaging omits it on recent setuptools (upstream also sets PYTHONPATH).
    sys.path.insert(0, str(ROOT / "vendor/libero"))
    config = ROOT / "runs/libero_config"
    config.mkdir(parents=True, exist_ok=True)
    datasets = ROOT / "datasets/libero"
    datasets.mkdir(parents=True, exist_ok=True)
    paths = {"benchmark_root": root, "bddl_files": root / "bddl_files",
             "init_states": root / "init_files", "assets": root / "assets",
             "datasets": datasets}
    (config / "config.yaml").write_text(yaml.safe_dump({k: str(v) for k, v in paths.items()}))
    os.environ["LIBERO_CONFIG_PATH"] = str(config)
    os.environ.setdefault("MPLCONFIGDIR", str(config / "matplotlib"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(config / "numba"))
    return paths


def configure_renderer():
    """Use MuJoCo's native offscreen CGL context on macOS.

    Robosuite 1.4.1 accepts 'cgl' but still creates a hidden GLFW window. Under
    mjpython that happens on the physics thread and AppKit aborts the process.
    This only replaces context creation; rendering, cameras and physics remain
    in the upstream simulator. Apply before constructing any environment.
    """
    if platform.system() != "Darwin":
        return "robosuite default"
    from mujoco.cgl import GLContext
    from robosuite.utils import binding_utils

    class RobosuiteCGLContext(GLContext):
        def __init__(self, max_width, max_height, device_id=-1):
            super().__init__(max_width, max_height)

    binding_utils.GLContext = RobosuiteCGLContext
    return "mujoco.cgl.GLContext"


def policy_observation(obs, prompt):
    """Same two cameras, 180-degree rotation, resize and state as OpenPI's client."""
    from openpi_client import image_tools
    images = {}
    for source, target in [("agentview_image", "observation/image"),
                           ("robot0_eye_in_hand_image", "observation/wrist_image")]:
        image = np.ascontiguousarray(obs[source][::-1, ::-1])
        images[target] = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, 224, 224))
    quat = np.asarray(obs["robot0_eef_quat"]).copy()
    w = np.clip(quat[3], -1., 1.)
    denominator = np.sqrt(1. - w * w)
    rotation = np.zeros(3) if math.isclose(denominator, 0.) else quat[:3] * 2 * math.acos(w) / denominator
    state = np.concatenate([obs["robot0_eef_pos"], rotation, obs["robot0_gripper_qpos"]])
    return {**images, "observation/state": state, "prompt": str(prompt)}
