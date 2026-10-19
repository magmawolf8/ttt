"""MuJoCo UR7e runtime. Policy observations contain no card ground truth."""
from pathlib import Path
import time
import json

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
DOWN = np.diag([1., -1., -1.])


class CardSim:
    def __init__(self, seed=0, profile="suction_upright"):
        self.model = mujoco.MjModel.from_xml_path(str(ROOT / "assets/scene.xml"))
        self.config = json.loads((ROOT/"assets/runtime_config.json").read_text())
        self.suction_config = self.config["suction"]
        self.profile_name = profile
        profiles=json.loads((ROOT/"assets/policy_profiles.json").read_text())
        selected=profiles[profile]
        self.policy_profile={**profiles.get(selected.get("inherit"),{}),**selected}
        if "external_camera_position_m" in self.policy_profile:
            ext=self.model.camera("external").id
            wrist=self.model.camera("wrist").id
            self.model.cam_pos[ext]=self.policy_profile["external_camera_position_m"]
            self.model.cam_quat[ext]=self.policy_profile["external_camera_quaternion_wxyz"]
            self.model.cam_fovy[ext]=self.policy_profile["external_camera_fovy_degrees"]
            self.model.cam_pos[wrist]=self.policy_profile["wrist_camera_position_in_cup_m"]
            self.model.cam_quat[wrist]=self.policy_profile["wrist_camera_quaternion_wxyz"]
        self.data = mujoco.MjData(self.model)
        self.tcp = self.model.site("tcp").id
        self.cup = self.model.body("cup").id
        self.card = self.model.body("card").id
        self.cup_geom = self.model.geom("cup_contact").id
        self.card_geom = self.model.geom("card_geom").id
        self.eq = self.model.equality("suction_weld").id
        self.card_qpos = self.model.jnt_qposadr[self.model.joint("card_free").id]
        self.rng = np.random.default_rng(seed)
        self.renderer = None
        self._renderers = {}
        self.suction = False
        self.suction_state = "off"
        self.dwell = 0.
        self.events = []
        self.interventions = []
        self.command_count = 0
        self.physics_observer = None
        self.reset(seed)

    def reset(self, seed=0):
        self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:6] = [0., -1.2, 1.8, -2.17, -1.57, 0.]
        yaw = self.rng.uniform(-np.pi, np.pi)
        self.data.qpos[self.card_qpos:self.card_qpos+7] = [self.rng.uniform(.40,.52),self.rng.uniform(-.10,.10),.00025,np.cos(yaw/2),0,0,np.sin(yaw/2)]
        self.data.ctrl[:] = self.data.qpos[:6]
        self.data.eq_active[self.eq] = False
        self.suction = False
        self.suction_state = "off"
        self.dwell = 0.
        self.events = []
        self.interventions = []
        self.command_count = 0
        mujoco.mj_forward(self.model,self.data)
        # Deterministic reset positioning, before any policy control begins.
        self.data.qpos[:6] = self.ik([.45,0,.23],DOWN, iterations=150)
        self.data.ctrl[:] = self.data.qpos[:6]
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model,self.data)
        self.step_physics(100)
        self.target_pos = self.position.copy()
        self.target_rot = self.rotation.copy()

    @property
    def position(self):
        return self.data.site_xpos[self.tcp].copy()

    @property
    def rotation(self):
        return self.data.site_xmat[self.tcp].reshape(3,3).copy()

    def ik(self, pos, rot, iterations=50):
        scratch = mujoco.MjData(self.model)
        scratch.qpos[:] = self.data.qpos
        jp = np.zeros((3,self.model.nv)); jr = np.zeros_like(jp)
        for _ in range(iterations):
            mujoco.mj_forward(self.model,scratch)
            current_r = scratch.site_xmat[self.tcp].reshape(3,3)
            error = np.r_[np.asarray(pos)-scratch.site_xpos[self.tcp], Rotation.from_matrix(rot @ current_r.T).as_rotvec()]
            if np.linalg.norm(error) < 1e-5:
                break
            mujoco.mj_jacSite(self.model,scratch,jp,jr,self.tcp)
            jac = np.vstack([jp[:,:6],jr[:,:6]])
            delta = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(6)*.0004,error)
            scratch.qpos[:6] += np.clip(delta,-.12,.12)
            scratch.qpos[:6] = np.clip(scratch.qpos[:6],self.model.jnt_range[:6,0],self.model.jnt_range[:6,1])
        return scratch.qpos[:6].copy()

    def set_suction(self, on):
        self.suction = bool(on)
        if not on:
            if self.data.eq_active[self.eq]:
                self._release("released")
            else:
                self.suction_state = "off"
            self.dwell = 0.
        elif not self.data.eq_active[self.eq]:
            self.suction_state = "attempting"

    def _release(self, reason):
        self.data.eq_active[self.eq] = False
        self.suction_state = reason
        self.dwell = 0.
        self.events.append({"time":float(self.data.time),"event":reason})

    def attachment_eligibility(self):
        # Used exclusively by the simulated suction mechanism, never the controller/policy.
        contact = any({int(c.geom1),int(c.geom2)}=={self.cup_geom,self.card_geom} and c.dist < .00015 for c in self.data.contact)
        local = self.data.xmat[self.card].reshape(3,3).T @ (self.position-self.data.xpos[self.card])
        radius = self.suction_config["cup_radius_m"]
        inside = abs(local[0]) < .0315-radius and abs(local[1]) < .044-radius
        normal = -self.rotation[:,2] @ self.data.xmat[self.card].reshape(3,3)[:,2]
        cup_velocity = np.zeros(6); card_velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,self.cup,cup_velocity,0)
        mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,self.card,card_velocity,0)
        slow = np.linalg.norm(cup_velocity[3:]-card_velocity[3:]) < self.suction_config["relative_speed_limit_m_s"]
        return contact and inside and normal > np.cos(np.deg2rad(self.suction_config["normal_tolerance_degrees"])) and slow and local[2] > -.00015

    def update_suction(self):
        if not self.suction:
            return
        if self.data.eq_active[self.eq]:
            rows = (self.data.efc_type==int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)) & (self.data.efc_id==self.eq)
            wrench = self.data.efc_force[rows]
            force = np.linalg.norm(wrench[:3]) if len(wrench) else 0.
            rel = self.data.xmat[self.cup].reshape(3,3).T @ (self.data.xpos[self.card]-self.data.xpos[self.cup])
            error = np.linalg.norm(rel-self.model.eq_data[self.eq,3:6])
            if force > self.suction_config["maximum_force_n"] or error > self.suction_config["maximum_constraint_error_m"]:
                self._release("failed_holding_limit")
            return
        if self.attachment_eligibility():
            self.dwell += self.model.opt.timestep
            if self.dwell >= self.suction_config["contact_dwell_s"]:
                relpos = self.data.xmat[self.cup].reshape(3,3).T @ (self.data.xpos[self.card]-self.data.xpos[self.cup])
                inv = self.data.xquat[self.cup].copy(); inv[1:] *= -1
                relquat = np.zeros(4)
                mujoco.mju_mulQuat(relquat,inv,self.data.xquat[self.card])
                self.model.eq_data[self.eq,3:6] = relpos
                self.model.eq_data[self.eq,6:10] = relquat
                self.data.eq_active[self.eq] = True
                self.suction_state = "attached"
                self.events.append({"time":float(self.data.time),"event":"attached"})
        else:
            self.dwell = 0.

    def step_physics(self, count=25):
        for _ in range(count):
            self.data.qfrc_applied[:6] = self.data.qfrc_bias[:6]
            mujoco.mj_step(self.model,self.data)
            self.update_suction()
            if self.physics_observer is not None:
                self.physics_observer(self)
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                raise RuntimeError("Nonfinite simulation state")

    def command_pose(self,pos,rot,suction):
        self.set_suction(suction)
        target = self.ik(pos,rot,iterations=35)
        # Joint target slew is explicit in each profile, at the 20 Hz command rate.
        joint_step=self.policy_profile["joint_target_limit_rad_per_step"]
        bounded_target = np.clip(target,self.data.ctrl-joint_step,self.data.ctrl+joint_step)
        if not np.allclose(target,bounded_target,atol=1e-9,rtol=0):
            self.interventions.append({"step":self.command_count,"kind":"joint_target_speed"})
        target = bounded_target
        self.last_joint_target = target.copy()
        self.data.ctrl[:] = target
        self.step_physics()
        self.command_count += 1

    def apply_policy_action(self, action):
        action = np.asarray(action,dtype=float)
        if action.shape != (7,) or not np.isfinite(action).all():
            self.interventions.append({"step":self.command_count,"kind":"invalid_command"})
            raise ValueError(f"Invalid policy action: {action}")
        bounded = np.clip(action,-1.,1.)
        if not np.array_equal(bounded,action):
            self.interventions.append({"step":self.command_count,"kind":"action_clipped","raw":action.tolist()})
        # LIBERO normalized relative commands, with profile-specific bounded steps.
        dpos = bounded[:3]*.05
        drot = bounded[3:6]*.5
        translation_step=self.policy_profile["translation_limit_m_per_step"]
        rotation_step=self.policy_profile["rotation_limit_rad_per_step"]
        if self.position[2]<self.policy_profile.get("near_table_height_m",-1):
            translation_step=min(translation_step,self.policy_profile["near_table_translation_limit_m_per_step"])
        if np.linalg.norm(dpos)>translation_step:
            self.interventions.append({"step":self.command_count,"kind":"translation_speed"})
            dpos *= translation_step/np.linalg.norm(dpos)
        if np.linalg.norm(drot)>rotation_step:
            self.interventions.append({"step":self.command_count,"kind":"rotation_speed"})
            drot *= rotation_step/np.linalg.norm(drot)
        requested = self.position + dpos
        pos = np.clip(requested,[.25,-.25,.0002],[.65,.25,.60])
        if not np.array_equal(requested,pos):
            self.interventions.append({"step":self.command_count,"kind":"workspace"})
        rot = Rotation.from_rotvec(drot).as_matrix() @ self.rotation
        if "maximum_cup_tilt_degrees" in self.policy_profile:
            # A tabletop suction tool must stay close to the known table normal.
            # Preserve policy yaw and translation; never use the card pose here.
            normal=rot[:,2]
            down=np.array([0.,0.,-1.])
            tilt=np.arccos(np.clip(normal@down,-1.,1.))
            limit=np.deg2rad(self.policy_profile["maximum_cup_tilt_degrees"])
            if tilt>limit:
                axis=np.cross(normal,down)
                norm=np.linalg.norm(axis)
                axis=axis/norm if norm>1e-9 else rot[:,0]
                rot=Rotation.from_rotvec(axis*(tilt-limit)).as_matrix()@rot
                self.interventions.append({"step":self.command_count,"kind":"cup_tilt"})
        self.last_policy_command = {"tcp_before_m":self.position.tolist(),"raw_action":action.tolist(),"requested_position_m":requested.tolist(),"bounded_position_m":pos.tolist(),"bounded_rotation_matrix":rot.tolist()}
        self.command_pose(pos,rot,bounded[6]>0)

    def render(self, camera="external", width=224, height=224):
        key=(width,height)
        if key not in self._renderers:
            self._renderers[key]=mujoco.Renderer(self.model,height,width)
        self.renderer=self._renderers[key]
        self.renderer.update_scene(self.data,camera=camera)
        return self.renderer.render().copy()

    def observation(self, prompt="Pick up the face-down card.",image_transform=None):
        # Experimental transfer: two virtual finger positions encode vacuum state.
        # This is a declared embodiment mismatch, not a learned suction skill.
        fingers = [0.,0.] if self.suction else [.04,-.04]
        policy_position = self.position + np.array(self.policy_profile["state_translation_m"])
        state = np.r_[policy_position,Rotation.from_matrix(self.rotation).as_rotvec(),fingers].astype(np.float32)
        transform=image_transform or self.policy_profile["image_transform"]
        resolution=self.policy_profile.get("policy_render_resolution",224)
        base=self.render(width=resolution,height=resolution)
        wrist=self.render("wrist",width=resolution,height=resolution)
        if resolution!=224:
            from openpi_client.image_tools import resize_with_pad
            base=resize_with_pad(base,224,224)
            wrist=resize_with_pad(wrist,224,224)
        def orient(image):
            if transform=="mirror_x":
                return np.ascontiguousarray(image[:,::-1])
            if transform=="rotate_180":
                return np.ascontiguousarray(image[::-1,::-1])
            return image
        return {"observation/state":state,"observation/image":orient(base),"observation/wrist_image":orient(wrist),"prompt":prompt}

    def close(self):
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()
        self.renderer=None


if __name__ == "__main__":
    from PIL import Image
    sim = CardSim()
    (ROOT/"runs").mkdir(exist_ok=True)
    Image.fromarray(sim.render(width=960,height=640)).save(ROOT/"runs/scene.png")
    Image.fromarray(sim.render("wrist")).save(ROOT/"runs/wrist.png")
    print({"tcp":sim.position.tolist(),"card":sim.data.xpos[sim.card].tolist(),"rotation":sim.rotation.tolist()})
    sim.close()
