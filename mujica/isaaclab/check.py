"""Offline validation; does not start Isaac Sim or step any physics."""
import importlib.metadata
import tempfile
import xml.etree.ElementTree as ET

from .assets import prepare_urdf, inertial_properties
from .settings import joint_contract, validate_settings
from .robots import robot_spec
from mujica.skills import skill_metadata
from mujica.terrain import HeightField, assign_terrain_columns
from mujica.rewards import reward_metadata


def check(settings):
    validate_settings(settings)
    spec = robot_spec(settings.asset.name)
    contract = joint_contract(robot=spec.name)
    if settings.terrain.mesh_type == "trimesh":
        terrain = HeightField(settings.terrain)
        columns = assign_terrain_columns(settings.env.num_envs, terrain.task_ids)
        tasks = terrain.task_ids[columns]
        task_counts = {name: int((tasks == i).sum()) for i, name in enumerate(skill_metadata()["skill_names"])}
        terrain_counts = {name: sum(terrain.terrain_names[col] == name for col in columns)
                          for name in dict.fromkeys(terrain.terrain_names)}
    else:
        task_counts = dict(flat_slope=settings.env.num_envs, discrete=0, stairs=0)
        terrain_counts = dict(flat=settings.env.num_envs)
    with tempfile.TemporaryDirectory(prefix="mujica_asset_check_") as tmp:
        path = prepare_urdf(cache_dir=tmp, robot=spec.name)
        root = ET.parse(path).getroot()
        total_mass = sum(inertial_properties(link)[0] for link in root.findall("link"))
        original = ET.parse(spec.urdf).getroot()
        original_mass = sum(inertial_properties(link)[0] for link in original.findall("link"))
        if abs(total_mass-original_mass) > 1e-8:
            raise ValueError("Fixed-link merging changed the total robot mass")
        versions = {}
        for name in ("isaaclab", "isaacsim", "torch", "numpy", "scipy"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = "not installed"
        return dict(**skill_metadata(), validation="offline_only", simulator="isaaclab", stage=settings.mujica.stage,
            robot=spec.name, original_urdf=str(spec.urdf), retained_bodies=len(root.findall("link")),
            total_mass_kg=total_mass, joint_names=list(spec.joints), joint_limits=contract,
            policy_observations=348, critic_observations=spec.critic_dim, actions=16,
            physics_dt=settings.sim.dt, control_dt=settings.sim.dt*settings.control.decimation,
            terrain=settings.terrain.mesh_type, terrain_grid=[settings.terrain.num_rows, settings.terrain.num_cols],
            task_environment_counts=task_counts, terrain_environment_counts=terrain_counts,
            reward_profile=reward_metadata(settings.rewards),
            installed_versions=versions)
