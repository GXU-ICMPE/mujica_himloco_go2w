"""Prepare the local URDF, preserving the Gym task's 19 rigid bodies.

Merge fixed attachments except the two head links before the Lab importer.
Geometry, mass, center of mass and inertia are transformed into the parent
frame. No simulator is needed. The original URDF and meshes stay untouched.
"""
import copy
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .settings import URDF_PATH, PROJECT_ROOT


def transform(origin):
    result = np.eye(4)
    if origin is not None:
        result[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        result[:3, :3] = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
    return result


def set_origin(element, matrix):
    origin = element.find("origin")
    if origin is None:
        origin = ET.SubElement(element, "origin")
    origin.set("xyz", " ".join(format(x, ".16g") for x in matrix[:3, 3]))
    origin.set("rpy", " ".join(format(x, ".16g") for x in Rotation.from_matrix(matrix[:3, :3]).as_euler("xyz")))


def inertial_properties(link):
    inertial = link.find("inertial")
    if inertial is None:
        return 0.0, np.zeros(3), np.zeros((3, 3))
    pose = transform(inertial.find("origin"))
    mass = float(inertial.find("mass").get("value"))
    entry = inertial.find("inertia")
    xx, xy, xz, yy, yz, zz = (float(entry.get(key)) for key in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"))
    inertia = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    return mass, pose[:3, 3], pose[:3, :3] @ inertia @ pose[:3, :3].T


def merge_inertia(parent, child, pose):
    ma, ca, ia = inertial_properties(parent)
    mb, cb, ib = inertial_properties(child)
    if mb == 0:
        return
    cb = pose[:3, :3] @ cb + pose[:3, 3]
    ib = pose[:3, :3] @ ib @ pose[:3, :3].T
    mass = ma + mb
    center = (ma*ca + mb*cb) / mass
    def shift(m, offset):
        return m * (np.dot(offset, offset)*np.eye(3) - np.outer(offset, offset))
    inertia = ia + ib + shift(ma, ca-center) + shift(mb, cb-center)
    old = parent.find("inertial")
    if old is not None:
        parent.remove(old)
    item = ET.SubElement(parent, "inertial")
    new_pose = np.eye(4)
    new_pose[:3, 3] = center
    set_origin(item, new_pose)
    ET.SubElement(item, "mass", value=str(mass))
    ET.SubElement(item, "inertia", **{key: str(inertia[i, j]) for key, i, j in (
        ("ixx", 0, 0), ("ixy", 0, 1), ("ixz", 0, 2), ("iyy", 1, 1), ("iyz", 1, 2), ("izz", 2, 2))})


def prepare_urdf(source=URDF_PATH, cache_dir=None):
    source = Path(source).resolve()
    # Include the conversion implementation so a corrected merger invalidates its cache.
    digest = hashlib.sha256(source.read_bytes() + Path(__file__).read_bytes()).hexdigest()[:16]
    output = Path(cache_dir or PROJECT_ROOT / ".cache/isaaclab") / digest / "go2w.urdf"
    if output.is_file():
        return output
    root = ET.parse(source).getroot()
    for mesh in root.findall(".//mesh"):
        mesh.set("filename", str((source.parent / mesh.get("filename")).resolve()))
    for joint in list(root.findall("joint")):
        if joint.get("type") != "fixed" or joint.get("dont_collapse") == "true":
            continue
        parent_name, child_name = joint.find("parent").get("link"), joint.find("child").get("link")
        parent = root.find(f"link[@name='{parent_name}']")
        child = root.find(f"link[@name='{child_name}']")
        pose = transform(joint.find("origin"))
        merge_inertia(parent, child, pose)
        for kind in ("visual", "collision"):
            for element in child.findall(kind):
                element = copy.deepcopy(element)
                set_origin(element, pose @ transform(element.find("origin")))
                parent.append(element)
        for descendant in root.findall("joint"):
            if descendant.find("parent").get("link") == child_name:
                descendant.find("parent").set("link", parent_name)
                set_origin(descendant, pose @ transform(descendant.find("origin")))
        root.remove(child)
        root.remove(joint)
    if len(root.findall("link")) != 19:
        raise ValueError("Prepared Go2W must retain 19 bodies including both head links")
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    temp = output.with_suffix(".tmp")
    ET.ElementTree(root).write(temp, encoding="utf-8", xml_declaration=True)
    temp.replace(output)
    return output
