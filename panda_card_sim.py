"""The original card/table scene with a native Panda arm and parallel gripper.

Card pose is used only for reset and evaluation. Control is robosuite OSC_POSE;
there is no attachment weld, task controller, or object information in observations.
"""
from copy import deepcopy
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from libero_runtime import ROOT, configure_paths, configure_renderer

configure_paths()
from robosuite import load_controller_config
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import EmptyArena
from robosuite.models.tasks import ManipulationTask


class PandaCardEnv(SingleArmEnv):
    def __init__(self, seed=42, gripper_type="default"):
        self.scene_seed = seed
        self.physics_observer = None
        self.render_backend = configure_renderer()
        super().__init__(robots="Panda", mount_types=None, gripper_types=gripper_type,
                         controller_configs=load_controller_config(default_controller="OSC_POSE"),
                         initialization_noise=None, has_renderer=False,
                         has_offscreen_renderer=True, use_camera_obs=True,
                         camera_names=["agentview", "robot0_eye_in_hand"],
                         camera_heights=224, camera_widths=224,
                         control_freq=20, horizon=1000, ignore_done=True, hard_reset=False)
        native_step = self.sim.step

        def observed_step():
            native_step()
            self._after_physics_step()
            if self.physics_observer is not None:
                self.physics_observer(self)

        self.sim.step = observed_step

    def _after_physics_step(self):
        """Optional physical tool update, before independent evaluation."""

    def _load_model(self):
        super()._load_model()
        self.robots[0].robot_model.set_base_xpos(np.zeros(3))
        original = ET.parse(ROOT / "assets/scene.xml").getroot()
        arena = EmptyArena()
        arena.worldbody.clear()
        arena.asset.clear()
        for name in ["texture", "material"]:
            for item in original.findall(f"asset/{name}"):
                item = deepcopy(item)
                if item.get("file"):
                    item.set("file", str(ROOT / "assets" / item.get("file")))
                arena.asset.append(item)
        for item in original.find("worldbody"):
            # Replace the UR7e body, keeping the original nonrobot scene.
            if item.tag == "body" and item.get("name") != "card":
                continue
            item = deepcopy(item)
            defaults = original.find("default/geom").attrib
            geoms = [item] if item.tag == "geom" else list(item.iter("geom"))
            for geom in geoms:
                for key, value in defaults.items():
                    if key not in geom.attrib:
                        geom.set(key, value)
                # Robosuite displays group 1; this does not disable collision.
                geom.set("group", "1")
            if item.tag == "body" and item.get("name") == "card":
                # Robosuite infers inertia only from group 0. Keep the same
                # box inertia explicitly when the single card geom is visible.
                geom = item.find("geom")
                half = np.fromstring(geom.get("size"), sep=" ")
                mass = float(geom.get("mass"))
                inertia = mass / 3 * (np.sum(half * half) - half * half)
                ET.SubElement(item, "inertial", pos="0 0 0", mass=str(mass),
                              diaginertia=" ".join(map(str, inertia)))
            if item.tag == "camera" and item.get("name") == "external":
                profile = json.loads((ROOT / "assets/policy_profiles.json").read_text())["libero"]
                item.attrib.clear()
                item.attrib.update(name="agentview", pos=" ".join(map(str, profile["external_camera_position_m"])),
                                   quat=" ".join(map(str, profile["external_camera_quaternion_wxyz"])),
                                   fovy=str(profile["external_camera_fovy_degrees"]))
            arena.worldbody.append(item)
        self.model = ManipulationTask(mujoco_arena=arena,
                                     mujoco_robots=[self.robots[0].robot_model])
        for tag in ["option", "visual"]:
            existing = self.model.root.find(tag)
            if existing is not None:
                self.model.root.remove(existing)
            self.model.root.append(deepcopy(original.find(tag)))

    def _setup_references(self):
        super()._setup_references()
        self.card_id = self.sim.model.body_name2id("card")
        self.card_geom_id = self.sim.model.geom_name2id("card_geom")
        self.table_geom_id = self.sim.model.geom_name2id("table")
        gripper = self.robots[0].gripper
        self.finger_groups = [
            {self.sim.model.geom_name2id(name) for name in gripper.important_geoms[key]}
            for key in ["left_finger", "right_finger"]
        ]

    def _reset_internal(self):
        super()._reset_internal()
        rng = np.random.default_rng(self.scene_seed)
        yaw = rng.uniform(-np.pi, np.pi)
        pose = [rng.uniform(.40, .52), rng.uniform(-.10, .10), .00025,
                np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.sim.data.set_joint_qpos("card_free", pose)
        # Match the original task's fixed initial tool position. No card pose
        # enters IK. The Panda grip site is the center between its fingertips.
        robot = self.robots[0]
        model, data = self.sim.model._model, self.sim.data._data
        qidx = robot._ref_joint_pos_indexes
        vidx = robot._ref_joint_vel_indexes
        jp = np.zeros((3, model.nv))
        jr = np.zeros_like(jp)
        target = np.array([.45, 0., .23])
        down = np.diag([1., -1., -1.])
        for _ in range(250):
            mujoco.mj_forward(model, data)
            rotation = data.site_xmat[robot.eef_site_id].reshape(3, 3)
            error = np.r_[target - data.site_xpos[robot.eef_site_id],
                          Rotation.from_matrix(down @ rotation.T).as_rotvec()]
            if np.linalg.norm(error) < 1e-6:
                break
            mujoco.mj_jacSite(model, data, jp, jr, robot.eef_site_id)
            jac = np.vstack([jp[:, vidx], jr[:, vidx]])
            delta = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(6) * .0001, error)
            data.qpos[qidx] += np.clip(delta, -.1, .1)
            joint_ids = [self.sim.model.joint_name2id(name) for name in robot.robot_model.joints]
            data.qpos[qidx] = np.clip(data.qpos[qidx], model.jnt_range[joint_ids, 0], model.jnt_range[joint_ids, 1])
        if np.linalg.norm(error) >= 1e-4:
            raise RuntimeError(f"Panda reset IK failed: {error}")
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        robot.controller.update_initial_joints(data.qpos[qidx].copy())
        robot.controller.update(force=True)
        robot.controller.reset_goal()

    def reward(self, action=None):
        return 0.0  # Scoring is performed by the independent physics observer.

    def _check_success(self):
        return False  # Never use a height-only reward as the pickup result.


class PandaCardDiagnostics:
    """Physical two-finger grasp + lowest card corner >=50mm for >=2 seconds."""
    def __init__(self, env):
        self.start_time = env.sim.data.time
        self.steps = self.contact_steps = self.bilateral_steps = self.table_contact_steps = 0
        self.minimum_finger_distance = float("inf")
        self.maximum_clearance = self.hold = self.maximum_hold = 0.
        self.closest = None

    def snapshot(self, env):
        model, data = env.sim.model._model, env.sim.data._data
        position = data.xpos[env.card_id]
        rotation = data.xmat[env.card_id].reshape(3, 3)
        extent = np.abs(rotation[2]) @ model.geom_size[env.card_geom_id]
        left, right = env.finger_groups
        card_contacts = set()
        table_contact = False
        for c in data.contact:
            pair = {int(c.geom1), int(c.geom2)}
            if env.card_geom_id in pair:
                card_contacts.update(pair - {env.card_geom_id})
            if env.table_geom_id in pair and (pair & (left | right)):
                table_contact = True
        distance = min(mujoco.mj_geomDistance(model, data, finger, env.card_geom_id, 2., None)
                       for finger in left | right)
        bilateral = bool(card_contacts & left) and bool(card_contacts & right)
        return {"t": float(data.time - self.start_time),
                "eef_m": data.site_xpos[env.robots[0].eef_site_id].tolist(),
                "card_m_evaluator_only": position.tolist(),
                "card_clearance_m": float(position[2] - extent),
                "finger_to_card_distance_m": float(distance),
                "left_finger_contact": bool(card_contacts & left),
                "right_finger_contact": bool(card_contacts & right),
                "bilateral_grasp": bilateral, "finger_table_contact": table_contact,
                "gripper_qpos": data.qpos[env.robots[0]._ref_gripper_joint_pos_indexes].tolist()}

    def physics_step(self, env):
        sample = self.snapshot(env)
        self.steps += 1
        self.contact_steps += sample["left_finger_contact"] or sample["right_finger_contact"]
        self.bilateral_steps += sample["bilateral_grasp"]
        self.table_contact_steps += sample["finger_table_contact"]
        if sample["finger_to_card_distance_m"] < self.minimum_finger_distance:
            self.minimum_finger_distance = sample["finger_to_card_distance_m"]
            self.closest = sample
        self.maximum_clearance = max(self.maximum_clearance, sample["card_clearance_m"])
        if sample["bilateral_grasp"] and sample["card_clearance_m"] >= .05:
            self.hold += env.sim.model.opt.timestep
            self.maximum_hold = max(self.maximum_hold, self.hold)
        else:
            self.hold = 0.

    def report(self):
        success = self.maximum_hold >= 2. - 1e-8
        failure = None if success else ("approach_or_alignment" if not self.contact_steps else
                   "contact_without_bilateral_grasp" if not self.bilateral_steps else "grasp_without_sustained_lift")
        return {"success": success, "failure_category": failure,
                "criterion": "bilateral finger contact and lowest card point >=0.05m continuously for 2s",
                "physics_steps": self.steps, "finger_card_contact_steps": self.contact_steps,
                "bilateral_grasp_steps": self.bilateral_steps, "finger_table_contact_steps": self.table_contact_steps,
                "minimum_finger_to_card_distance_m": self.minimum_finger_distance,
                "maximum_card_clearance_m": self.maximum_clearance,
                "maximum_continuous_valid_hold_s": self.maximum_hold, "closest_approach": self.closest}
