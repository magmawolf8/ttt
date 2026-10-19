"""Build MJCF from the manufacturer's UR7e parameters and shared collision meshes."""
import json
import math
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation
import yaml

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "vendor/ur_description"
OUT = ROOT / "assets"
JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
LINKS = ["shoulder", "upper_arm", "forearm", "wrist_1", "wrist_2", "wrist_3"]


class Loader(yaml.SafeLoader):
    pass


Loader.add_constructor("!degrees", lambda loader, node: math.radians(float(loader.construct_scalar(node))))


def quat(rpy):
    q = Rotation.from_euler("xyz", rpy).as_quat()
    return [q[3], *q[:3]]


def vec(values):
    return " ".join(f"{float(v):.12g}" for v in values)


def add(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})


def main():
    OUT.mkdir(exist_ok=True)
    def load(name):
        return yaml.load((SOURCE / "config/ur7e" / name).read_text(), Loader=Loader)
    kin = load("default_kinematics.yaml")["kinematics"]
    phys = load("physical_parameters.yaml")["inertia_parameters"]
    visual = load("visual_parameters.yaml")["mesh_files"]
    limits = load("joint_limits.yaml")["joint_limits"]
    root = ET.Element("mujoco", model="UR7e card manipulation")
    add(root, "compiler", angle="radian", autolimits="true", meshdir="../vendor/ur_description/meshes/ur5e/collision")
    add(root, "option", timestep="0.002", integrator="implicitfast", cone="elliptic", iterations="80", gravity="0 0 -9.81")
    defaults = add(root, "default")
    add(defaults, "joint", armature="0.1", damping="1")
    add(defaults, "geom", friction="0.8 0.01 0.0001", solref="0.01 1", solimp="0.95 0.99 0.0001", margin="0.00005")
    vis = add(root, "visual")
    add(vis, "global", offwidth="1280", offheight="720")
    add(vis, "headlight", ambient="0.5 0.5 0.5", diffuse="0.6 0.6 0.6")
    asset = add(root, "asset")
    image = Image.new("RGB", (252, 352), "#1e448d")
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 243, 343), outline="white", width=6)
    for y in range(20, 340, 18):
        for x in range(20, 240, 18):
            draw.polygon([(x, y-5), (x+5, y), (x, y+5), (x-5, y)], fill="#83a4e0")
    image.save(OUT / "card_back.png")
    add(asset, "texture", name="card_back", type="2d", file="card_back.png")
    add(asset, "material", name="card_mat", texture="card_back", texuniform="false", reflectance="0.05")
    for link in ["base", *LINKS]:
        stem = {"upper_arm": "upperarm", "wrist_1": "wrist1", "wrist_2": "wrist2", "wrist_3": "wrist3"}.get(link, link)
        add(asset, "mesh", name=link+"_mesh", file=stem+".stl")
    world = add(root, "worldbody")
    add(world, "light", pos="0 -0.5 2", dir="0 0 -1", diffuse="0.9 0.9 0.9")
    add(world, "geom", name="floor", type="plane", size="2 2 0.1", pos="0 0 -0.8", rgba="0.19 0.20 0.23 1")
    add(world, "geom", name="table", type="box", pos="0.35 0 -0.035", size="0.65 0.5 0.035", rgba="0.38 0.28 0.18 1")
    target = np.array([0.35, 0, 0.1])
    eye = np.array([1.15, -1.05, 1.15])
    z = (eye-target) / np.linalg.norm(eye-target)
    x = np.cross([0,0,1], z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    add(world, "camera", name="external", pos=vec(eye), xyaxes=vec([*x,*y]), fovy="48")
    base = add(world, "body", name="base", quat="0 0 0 1")
    def mesh(body, name):
        offset = visual[name]["mesh_offset"]
        color = "0.45 0.66 0.8 1" if name in ["shoulder", "wrist_1", "wrist_2"] else "0.75 0.77 0.78 1"
        add(body, "geom", name=name+"_collision", type="mesh", mesh=name+"_mesh", pos=vec([offset[a] for a in "xyz"]), quat=vec(quat([offset[a] for a in ["roll","pitch","yaw"]])), rgba=color)
    mesh(base, "base")
    parent = base
    for link, joint in zip(LINKS, JOINTS):
        k = kin[link]
        body = add(parent, "body", name=link+"_link", pos=vec([k[a] for a in "xyz"]), quat=vec(quat([k[a] for a in ["roll","pitch","yaw"]])))
        lim = limits[joint]
        add(body, "joint", name=joint, axis="0 0 1", range=vec([lim["min_position"], lim["max_position"]]))
        com = phys["center_of_mass"][link+"_cog"]
        r = phys["rotation"][link]
        tensor = phys["tensor"][link]
        # MuJoCo fullinertia is expressed in body coordinates, so rotate the URDF inertia tensor.
        inertia = np.array([[tensor["ixx"],tensor["ixy"],tensor["ixz"]],[tensor["ixy"],tensor["iyy"],tensor["iyz"]],[tensor["ixz"],tensor["iyz"],tensor["izz"]]])
        rot = Rotation.from_euler("xyz",[r[a] for a in ["roll","pitch","yaw"]]).as_matrix()
        inertia = rot @ inertia @ rot.T
        add(body, "inertial", pos=vec([com[a] for a in "xyz"]), mass=phys[link+"_mass"], fullinertia=vec([inertia[0,0],inertia[1,1],inertia[2,2],inertia[0,1],inertia[0,2],inertia[1,2]]))
        mesh(body, link)
        parent = body
    # tool0 coincides with wrist_3 in the official chain; cup extends along +Z.
    cup = add(parent,"body",name="cup",pos="0 0 0.055")
    add(cup,"geom",name="cup_stem",type="cylinder",size="0.009 0.025",pos="0 0 -0.025",mass="0.07",rgba="0.25 0.25 0.28 1")
    add(cup,"geom",name="cup_contact",type="cylinder",size="0.009 0.002",mass="0.01",rgba="0.12 0.12 0.14 1",friction="1 0.01 0.0001")
    add(cup,"site",name="tcp",pos="0 0 0.002",size="0.002",rgba="1 0.3 0.2 1")
    add(cup,"camera",name="wrist",pos="0.045 0 -0.025",xyaxes="1 0 0 0 -1 0",fovy="75")
    card = add(world,"body",name="card",pos="0.45 0 0.0002")
    add(card,"freejoint",name="card_free")
    add(card,"geom",name="card_geom",type="box",size="0.0315 0.044 0.00015",mass="0.0018",material="card_mat",condim="4")
    act = add(root,"actuator")
    for index,joint in enumerate(JOINTS):
        lim = limits[joint]
        add(act,"position",name=joint+"_servo",joint=joint,kp="2000" if index<3 else "500",kv="200" if index<3 else "40",ctrlrange=vec([lim["min_position"],lim["max_position"]]),forcerange=vec([-lim["max_effort"],lim["max_effort"]]))
    eq = add(root,"equality")
    add(eq,"weld",name="suction_weld",body1="cup",body2="card",active="false",solref="0.006 1",solimp="0.95 0.99 0.001",torquescale="0.02")
    # Exclude only adjacent rigid links; all nonadjacent arm and table collisions remain enabled.
    contacts = add(root,"contact")
    names = ["base",*[link+"_link" for link in LINKS],"cup"]
    for a,b in zip(names,names[1:]):
        add(contacts,"exclude",body1=a,body2=b)
    ET.indent(root)
    ET.ElementTree(root).write(OUT / "scene.xml",encoding="unicode")
    manifest = {"robot":"UR7e", "description_revision":subprocess.check_output(["git","-C",str(SOURCE),"rev-parse","HEAD"],text=True).strip(),"physics_hz":500,"control_hz":20,"card_dimensions_m":[0.063,0.088,0.0003],"card_mass_kg":0.0018,"cup_diameter_m":0.018,"suction_model":"conditional weld, finite force and displacement break","mesh_note":"UR7e manufacturer config explicitly shares UR5e meshes"}
    (OUT/"scene_manifest.json").write_text(json.dumps(manifest,indent=2))
    print(OUT/"scene.xml")


if __name__ == "__main__":
    main()
