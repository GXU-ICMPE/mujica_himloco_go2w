"""Robot-specific task settings and policy interfaces, without simulator imports."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from mujica.skills import TASK_FAMILY, TASK_CONTRACT_VERSION, validate_skill_metadata
from mujica.terrain import validate_terrain
from mujica.rewards import validate_reward_settings
from .robots import robot_spec

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


def default_settings(robot="go2w"):
    data = json.loads(Path(__file__).with_name("defaults.json").read_text())
    robot_spec(robot)
    if robot == "x5":
        from .x5_config import configure_x5
        configure_x5(data)
    return settings_from_dict(data)


def joint_contract(path=None, robot="go2w"):
    spec = robot_spec(robot)
    path = Path(path or spec.urdf)
    root = ET.parse(path).getroot()
    joints = {j.get("name"): j for j in root.findall("joint") if j.get("type") != "fixed"}
    if set(joints) != set(spec.joints):
        raise ValueError(f"URDF actuated joints do not match the {robot} policy interface")
    result = {}
    for name in spec.joints:
        joint = joints[name]
        limit = joint.find("limit")
        result[name] = {key: float(limit.get(key)) for key in ("effort", "velocity")}
        # Continuous wheel angles are excluded from position-limit penalties.
        result[name].update(lower=float(limit.get("lower", "-3.141592653589793")),
                            upper=float(limit.get("upper", "3.141592653589793")))
    for mesh in root.findall(".//mesh"):
        if not (Path(path).parent / mesh.get("filename")).is_file():
            raise FileNotFoundError(mesh.get("filename"))
    return result


def joint_indices(simulator_names, robot="go2w"):
    names = robot_spec(robot).joints
    if len(simulator_names) != 16 or set(simulator_names) != set(names):
        raise ValueError(f"Imported articulation must contain exactly the 16 named {robot} joints")
    return [simulator_names.index(name) for name in names]


def validate_settings(settings):
    if settings.mujica.stage not in ("s1", "s2"):
        raise ValueError("stage must be s1 or s2")
    spec = robot_spec(settings.asset.name)
    if (settings.rewards.get('profile') == 'x5_v8_mujica_v1') != (spec.name == 'x5'):
        raise ValueError('The X5 v8 reward/control profile must be paired with the X5 robot')
    if set(settings.init_state.default_joint_angles) != set(spec.joints):
        raise ValueError(f'Default joint angles must follow the {spec.name} named joint contract')
    if (settings.env.num_observations, settings.env.num_privileged_obs, settings.env.num_actions) != (348, spec.critic_dim, 16):
        raise ValueError(f"MUJICA {spec.name} requires observations=348, critic={spec.critic_dim}, actions=16")
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


def validate_checkpoint_backend(saved, *, resume, robot=None):
    metadata = saved.get("metadata", {})
    if resume and metadata.get("simulator") != "isaaclab":
        raise ValueError("--resume requires an Isaac Lab checkpoint; a Gym checkpoint is not a continuation of Lab physics")
    robot = robot or metadata.get("environment_config", {}).get("asset", {}).get("name", "go2w")
    if metadata.get("joint_names") != list(robot_spec(robot).joints):
        raise ValueError("Checkpoint joint order is missing or differs from the Isaac Lab policy ABI")
    validate_skill_metadata(metadata)
