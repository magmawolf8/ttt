"""Ground-truth evaluation and video annotation, outside the policy observation path."""
from collections import Counter
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class EpisodeDiagnostics:
    def __init__(self,sim,out):
        self.start_time=float(sim.data.time)
        self.contact_steps=0
        self.eligible_steps=0
        self.suction_steps=0
        self.attachment_steps=0
        self.footprint_steps=0
        self.minimum_distance=float("inf")
        self.closest=None
        self.minimum_tcp_z=float("inf")
        self.maximum_clearance=0.
        self.hold=0.
        self.maximum_hold=0.
        self.steps=0
        self.records=[]
        self.visibility=[]
        self.out=Path(out)

    def snapshot(self,sim):
        tcp=sim.position
        card=sim.data.xpos[sim.card].copy()
        rotation=sim.data.xmat[sim.card].reshape(3,3)
        local=rotation.T@(tcp-card)
        nearest=np.clip(local,[-.0315,-.044,-.00015],[.0315,.044,.00015])
        distance=float(np.linalg.norm(local-nearest))
        clearance=float(card[2]-np.abs(rotation[2])@np.array([.0315,.044,.00015]))
        contact=any({int(c.geom1),int(c.geom2)}=={sim.cup_geom,sim.card_geom} for c in sim.data.contact)
        normal_dot=float(np.clip(-sim.rotation[:,2]@rotation[:,2],-1,1))
        radius=sim.suction_config["cup_radius_m"]
        inside=abs(local[0])<.0315-radius and abs(local[1])<.044-radius
        return {"t":float(sim.data.time-self.start_time),"tcp_m":tcp.tolist(),"card_m":card.tolist(),"tcp_card_local_m":local.tolist(),"tcp_to_card_surface_m":distance,"card_clearance_m":clearance,"cup_card_contact":contact,"cup_inside_card_footprint":bool(inside),"normal_error_degrees":float(np.rad2deg(np.arccos(normal_dot))),"suction":sim.suction,"attached":bool(sim.data.eq_active[sim.eq]),"suction_state":sim.suction_state,"contact_dwell_s":float(sim.dwell)}

    def physics_step(self,sim):
        sample=self.snapshot(sim)
        self.steps+=1
        self.contact_steps+=sample["cup_card_contact"]
        self.eligible_steps+=bool(sim.attachment_eligibility())
        self.suction_steps+=sim.suction
        self.attachment_steps+=sample["attached"]
        self.footprint_steps+=sample["cup_inside_card_footprint"]
        self.minimum_tcp_z=min(self.minimum_tcp_z,sample["tcp_m"][2])
        self.maximum_clearance=max(self.maximum_clearance,sample["card_clearance_m"])
        if sample["tcp_to_card_surface_m"]<self.minimum_distance:
            self.minimum_distance=sample["tcp_to_card_surface_m"]
            self.closest=sample
        if sample["attached"] and sample["card_clearance_m"]>=.05:
            self.hold+=sim.model.opt.timestep
            self.maximum_hold=max(self.maximum_hold,self.hold)
        else:
            self.hold=0.

    def record_command(self,sim,chunk,index):
        sample=self.snapshot(sim)
        sample.update(chunk=chunk,action_in_chunk=index,command=sim.last_policy_command,joint_positions_rad=sim.data.qpos[:6].tolist(),joint_targets_rad=sim.data.ctrl.tolist())
        self.records.append(sample)
        return sample

    def record_visibility(self,sim,chunk):
        counts={"chunk":chunk,"t":float(sim.data.time-self.start_time)}
        for camera in ["external","wrist"]:
            sim.render(camera)
            sim.renderer.enable_segmentation_rendering()
            try:
                segmentation=sim.renderer.render()
                mask=(segmentation[...,0]==sim.card_geom)&(segmentation[...,1]==int(mujoco.mjtObj.mjOBJ_GEOM))
                counts[camera+"_card_pixels"]=int(mask.sum())
            finally:
                sim.renderer.disable_segmentation_rendering()
        self.visibility.append(counts)
        return counts

    def save(self,sim):
        self.out.joinpath("telemetry.jsonl").write_text("\n".join(json.dumps(row) for row in self.records)+"\n")
        self.out.joinpath("visibility.json").write_text(json.dumps(self.visibility,indent=2))
        attached=self.attachment_steps>0
        success=self.maximum_hold>=2.-1e-8
        failure=None if success else ("approach_or_alignment" if not self.contact_steps else "no_attachment" if not attached else "drop_or_insufficient_hold")
        report={"success":success,"failure_category":failure,"physics_steps":self.steps,"duration_simulated_s":self.steps*sim.model.opt.timestep,"minimum_tcp_to_card_surface_m":self.minimum_distance,"closest_approach":self.closest,"minimum_tcp_height_m":self.minimum_tcp_z,"maximum_card_clearance_m":self.maximum_clearance,"maximum_continuous_valid_hold_s":self.maximum_hold,"cup_card_contact_steps":self.contact_steps,"attachment_eligible_steps":self.eligible_steps,"attached_steps":self.attachment_steps,"suction_on_fraction":self.suction_steps/max(1,self.steps),"cup_inside_footprint_fraction":self.footprint_steps/max(1,self.steps),"controller_interventions":dict(Counter(row["kind"] for row in sim.interventions)),"visibility":self.visibility}
        self.out.joinpath("diagnostics.json").write_text(json.dumps(report,indent=2))
        return report


def annotate_video_frame(external,base,wrist,sample,chunk_latency,duration_s=20,image_transform="native",policy_label="pi0.5"):
    frame=Image.new("RGB",(960,608),"#111821")
    frame.paste(Image.fromarray(external),(16,80))
    frame.paste(Image.fromarray(base),(720,80))
    frame.paste(Image.fromarray(wrist),(720,336))
    draw=ImageDraw.Draw(frame)
    font_path="C:/Windows/Fonts/segoeui.ttf"
    try:
        large=ImageFont.truetype(font_path,23)
        small=ImageFont.truetype(font_path,17)
    except OSError:
        large=small=ImageFont.load_default()
    draw.text((16,9),f"{policy_label} | UR7e card pickup",font=large,fill="white")
    status=f"t = {sample['t']:05.2f} / {duration_s:.2f} s     Vacuum: {'ON' if sample['suction'] else 'OFF'}     Attached: {'YES' if sample['attached'] else 'NO'}"
    draw.text((16,43),status,font=small,fill="#91d9ff")
    draw.text((720,57),"External policy camera",font=small,fill="white")
    draw.text((720,311),"Wrist policy camera",font=small,fill="white")
    draw.text((16,566),f"Cup-card distance: {sample['tcp_to_card_surface_m']*1000:.0f} mm  |  Inference: {chunk_latency:.2f} s/chunk",font=small,fill="white")
    draw.text((16,586),f"Inputs: {image_transform} | Simulated-time playback; physics pauses for inference.",font=small,fill="#a9b3c2")
    return np.asarray(frame)
