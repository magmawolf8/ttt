"""Original finite-strength suction tool on Panda, retaining a 7-value action API."""
from copy import deepcopy
import json
import xml.etree.ElementTree as ET

import numpy as np

from card_sim import CardSim
from episode_diagnostics import EpisodeDiagnostics
from libero_runtime import ROOT, policy_observation
from panda_card_sim import PandaCardEnv
from robosuite.models.grippers import GRIPPER_MAPPING, GripperModel


def build_suction_gripper():
    """Copy the original cup; put its face at Panda's original 97mm grip site."""
    original = ET.parse(ROOT / "assets/scene.xml").getroot()
    cup = deepcopy(original.find(".//body[@name='cup']"))
    cup.set("name", "eef")
    cup.set("pos", "0 0 0.095")
    for camera in list(cup.findall("camera")):
        cup.remove(camera)  # Keep Panda's native wrist camera unchanged.
    tcp = cup.find("site[@name='tcp']")
    tcp.set("name", "grip_site")
    tcp.set("rgba", "1 0 0 0")
    ET.SubElement(cup, "site", name="grip_site_cylinder", pos="0 0 0.002",
                  size="0.002 0.01", type="cylinder", rgba="0 1 0 0", group="1")
    for geom in list(cup.findall("geom")):
        for key, value in original.find("default/geom").attrib.items():
            geom.attrib.setdefault(key, value)
        geom.set("group", "0")
        visual = deepcopy(geom)
        visual.set("name", geom.get("name") + "_visual")
        visual.attrib.update(group="1", contype="0", conaffinity="0", mass="0")
        cup.append(visual)
    root = ET.Element("mujoco", model="original suction cup with Panda mounting adapter")
    world = ET.SubElement(root, "worldbody")
    mount = ET.SubElement(world, "body", name="right_gripper", quat="0.707107 0 0 -0.707107")
    # Mounting collar bridges the shorter original cup to the trained TCP.
    ET.SubElement(mount, "geom", name="mount_collar", type="cylinder", pos="0 0 0.0225",
                  size="0.012 0.0225", mass="0.02", group="0")
    ET.SubElement(mount, "geom", name="mount_collar_visual", type="cylinder", pos="0 0 0.0225",
                  size="0.012 0.0225", mass="0", group="1", contype="0", conaffinity="0", rgba="0.5 0.5 0.5 1")
    ET.SubElement(mount, "site", name="ft_frame", pos="0 0 0", size="0.001", rgba="0 0 0 0")
    mount.append(cup)
    sensors = ET.SubElement(root, "sensor")
    ET.SubElement(sensors, "force", name="force_ee", site="ft_frame")
    ET.SubElement(sensors, "torque", name="torque_ee", site="ft_frame")
    path = ROOT / "runs/panda_suction_assets/gripper.xml"
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="unicode")
    return path


class OriginalSuctionGripper(GripperModel):
    def __init__(self, idn=0):
        super().__init__(str(build_suction_gripper()), idn=idn)

    @property
    def dof(self):
        return 1  # One logical vacuum command, not an extra robot joint.

    @property
    def init_qpos(self):
        return np.empty(0)

    def format_action(self, action):
        if np.asarray(action).shape != (1,) or not np.isfinite(action).all():
            raise ValueError("Suction expects one finite gripper command")
        self.current_action = np.asarray(action).copy()
        return np.empty(0)  # Vacuum has no joint actuator; the environment handles it.


GRIPPER_MAPPING["OriginalSuctionCup"] = OriginalSuctionGripper


class OriginalSuctionMechanism:
    """Bind the exact existing CardSim suction methods to Panda's model/data.

    This narrow adapter deliberately reuses the original implementations rather
    than copying or weakening their seal/holding rules. No arm control is reused.
    """
    position = CardSim.position
    rotation = CardSim.rotation
    set_suction = CardSim.set_suction
    _release = CardSim._release
    attachment_eligibility = CardSim.attachment_eligibility
    update_suction = CardSim.update_suction

    def __init__(self, env):
        self.model, self.data = env.sim.model._model, env.sim.data._data
        gripper = env.robots[0].gripper
        self.tcp = env.robots[0].eef_site_id
        self.cup = self.model.body(gripper.correct_naming("eef")).id
        self.cup_geom = self.model.geom(gripper.correct_naming("cup_contact")).id
        self.card, self.card_geom = env.card_id, env.card_geom_id
        self.eq = self.model.equality("suction_weld").id
        self.suction_config = json.loads((ROOT / "assets/runtime_config.json").read_text())["suction"]
        self.data.eq_active[self.eq] = False
        self.suction = False
        self.suction_state = "off"
        self.dwell = 0.
        self.events = []


class PandaSuctionEnv(PandaCardEnv):
    def __init__(self, seed=42):
        self.suction_tool = None
        super().__init__(seed, gripper_type="OriginalSuctionCup")

    def _load_model(self):
        super()._load_model()
        original = ET.parse(ROOT / "assets/scene.xml").getroot()
        weld = deepcopy(original.find("equality/weld[@name='suction_weld']"))
        weld.set("body1", self.robots[0].gripper.correct_naming("eef"))
        self.model.equality.append(weld)

    def _reset_internal(self):
        super()._reset_internal()
        self.suction_tool = OriginalSuctionMechanism(self)

    def _pre_action(self, action, policy_step=False):
        action = np.asarray(action)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("Expected finite [dx,dy,dz,rx,ry,rz,gripper] action")
        # Same sign as upstream close/open and the original UR7e vacuum mapping.
        self.suction_tool.set_suction(action[6] > 0)
        super()._pre_action(action, policy_step)

    def _after_physics_step(self):
        self.suction_tool.update_suction()

    def observation_for_policy(self, obs, prompt):
        # Preserve the existing 8-value LIBERO interface without fake joints.
        # These are vacuum command indicators, not seal feedback; this specific
        # pretrained checkpoint ignores state, as recorded in its manifest.
        fingers = np.array([0., 0.] if self.suction_tool.suction else [.04, -.04])
        return policy_observation({**obs, "robot0_gripper_qpos": fingers}, prompt)


class PandaSuctionDiagnostics:
    """Same evaluator geometry and hold criterion as the original suction scene."""
    def __init__(self, env):
        self.start_time = env.sim.data.time
        self.steps = self.contact_steps = self.attachment_steps = self.eligible_steps = 0
        self.minimum_distance = float("inf")
        self.maximum_clearance = self.hold = self.maximum_hold = 0.
        self.closest = None

    def snapshot(self, env):
        sample = EpisodeDiagnostics.snapshot(self, env.suction_tool)
        sample["eef_m"] = sample.pop("tcp_m")
        sample["card_m_evaluator_only"] = sample.pop("card_m")
        return sample

    def physics_step(self, env):
        sample = self.snapshot(env)
        self.steps += 1
        self.contact_steps += sample["cup_card_contact"]
        self.attachment_steps += sample["attached"]
        self.eligible_steps += bool(env.suction_tool.attachment_eligibility())
        if sample["tcp_to_card_surface_m"] < self.minimum_distance:
            self.minimum_distance = sample["tcp_to_card_surface_m"]
            self.closest = sample
        self.maximum_clearance = max(self.maximum_clearance, sample["card_clearance_m"])
        if sample["attached"] and sample["card_clearance_m"] >= .05:
            self.hold += env.sim.model.opt.timestep
            self.maximum_hold = max(self.maximum_hold, self.hold)
        else:
            self.hold = 0.

    def report(self):
        success = self.maximum_hold >= 2. - 1e-8
        failure = None if success else ("approach_or_alignment" if not self.contact_steps else
                  "contact_without_suction_seal" if not self.attachment_steps else "drop_or_insufficient_hold")
        return {"success": success, "failure_category": failure,
                "criterion": "original suction attachment and lowest card point >=0.05m continuously for 2s",
                "physics_steps": self.steps, "cup_card_contact_steps": self.contact_steps,
                "attachment_eligible_steps": self.eligible_steps, "attached_steps": self.attachment_steps,
                "minimum_tcp_to_card_surface_m": self.minimum_distance,
                "maximum_card_clearance_m": self.maximum_clearance,
                "maximum_continuous_valid_hold_s": self.maximum_hold, "closest_approach": self.closest}
