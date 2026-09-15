"""Frozen Gym task parameters and the policy interface, without Gym imports."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from mujica.skills import TASK_FAMILY, TASK_CONTRACT_VERSION, validate_skill_metadata
from mujica.terrain import validate_terrain
from mujica.rewards import validate_reward_settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
URDF_PATH = PROJECT_ROOT / "resources/robots/go2w/urdf/go2w.urdf"
# A policy ABI, independent of USD/PhysX breadth-first articulation ordering.
JOINT_NAMES = tuple(f"{leg}_{segment}_joint" for leg in ("FL", "FR", "RL", "RR")
                    for segment in ("hip", "thigh", "calf", "foot"))


class Settings(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    __setattr__ = dict.__setitem__


def settings_from_dict(data):
    return Settings({key: settings_from_dict(value) if isinstance(value, dict) else value
                     for key, value in data.items()})


def default_settings():
    return settings_from_dict(json.loads(Path(__file__).with_name("defaults.json").read_text()))


def joint_contract(path=URDF_PATH):
    root = ET.parse(path).getroot()
    joints = {j.get("name"): j for j in root.findall("joint") if j.get("type") != "fixed"}
    if set(joints) != set(JOINT_NAMES):
        raise ValueError("URDF actuated joints do not match the Go2W policy interface")
    result = {}
    for name in JOINT_NAMES:
        joint = joints[name]
        limit = joint.find("limit")
        result[name] = {key: float(limit.get(key)) for key in ("lower", "upper", "effort", "velocity")}
    for mesh in root.findall(".//mesh"):
        if not (Path(path).parent / mesh.get("filename")).is_file():
            raise FileNotFoundError(mesh.get("filename"))
    return result


def joint_indices(simulator_names):
    if len(simulator_names) != 16 or set(simulator_names) != set(JOINT_NAMES):
        raise ValueError("Imported articulation must contain exactly the 16 named Go2W joints")
    return [simulator_names.index(name) for name in JOINT_NAMES]


def validate_settings(settings):
    if settings.mujica.stage not in ("s1", "s2"):
        raise ValueError("stage must be s1 or s2")
    if (settings.env.num_observations, settings.env.num_privileged_obs, settings.env.num_actions) != (348, 270, 16):
        raise ValueError("MUJICA requires observations=348, critic=270, actions=16")
    if not settings.terrain.measure_heights or len(settings.terrain.measured_points_x)*len(settings.terrain.measured_points_y) != 187:
        raise ValueError("MUJICA requires the 17 x 11 height grid")
    if settings.commands.heading_command or settings.commands.curriculum:
        raise ValueError("This backend preserves MUJICA's direct velocity commands and terrain-only curriculum")
    if settings.terrain.mesh_type not in ("plane", "trimesh"):
        raise ValueError("Isaac Lab backend supports plane or trimesh")
    if settings.control.control_type != "P" or settings.sim.substeps != 1:
        raise ValueError("Use explicit mixed PD control and one physics substep")
    if settings.sim.dt <= 0 or settings.control.decimation < 1 or settings.env.num_envs < 1:
        raise ValueError("Time steps and environment count must be positive")
    if settings.mujica.task_family != TASK_FAMILY or settings.mujica.task_contract_version != TASK_CONTRACT_VERSION:
        raise ValueError("Environment settings must describe the current terrain locomotion tasks")
    t = settings.terrain
    validate_terrain(t)
    validate_reward_settings(settings.rewards, t)
    if t.max_init_terrain_level < 0:
        raise ValueError("Initial terrain level must be non-negative")
    if not 0 < settings.mujica.terrain_edge_margin < min(t.terrain_length, t.terrain_width)/2:
        raise ValueError("terrain_edge_margin must lie inside the tile")
    if min(settings.commands.resampling_time, settings.env.episode_length_s,
           settings.domain_rand.disturbance_interval, settings.domain_rand.push_interval_s) <= 0:
        raise ValueError("Command, episode and disturbance intervals must be positive")


def validate_checkpoint_backend(saved, *, resume):
    metadata = saved.get("metadata", {})
    if resume and metadata.get("simulator") != "isaaclab":
        raise ValueError("--resume requires an Isaac Lab checkpoint; a Gym checkpoint is not a continuation of Lab physics")
    if metadata.get("joint_names") != list(JOINT_NAMES):
        raise ValueError("Checkpoint joint order is missing or differs from the Isaac Lab policy ABI")
    validate_skill_metadata(metadata)
