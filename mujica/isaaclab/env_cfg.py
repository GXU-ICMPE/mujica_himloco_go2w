"""Native DirectRLEnv configuration. Import after launching Isaac Sim."""
import json
from dataclasses import MISSING

import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from mujica.motor import DCMotorLimiter
from .assets import prepare_urdf
from .settings import joint_contract, validate_settings
from .robots import robot_spec


def spawn_x5_from_urdf(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Apply contact offsets on editable colliders before environment cloning.

    Adapted from X5loco (Copyright 2026 Robot-Nav, Apache-2.0);
    see third_party/X5loco-LICENSE. Added kwargs for the MUJICA spawner.
    """
    from isaaclab.sim.utils import make_uninstanceable
    prim = sim_utils.spawn_from_urdf(prim_path, cfg.replace(collision_props=None),
                                   translation=translation, orientation=orientation, **kwargs)
    # The outer clone wrapper supplies the concrete source path here.
    for body in prim.GetChildren():
        collisions = body.GetChild("collisions")
        if collisions.IsValid():
            make_uninstanceable(str(collisions.GetPath()))
    if cfg.collision_props is not None:
        sim_utils.modify_collision_properties(prim_path, cfg.collision_props)
    return prim


@configclass
class MUJICAEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 20.0
    action_space = 16
    observation_space = 348
    state_space = 270
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=3.0, replicate_physics=True)
    robot: ArticulationCfg = MISSING
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", update_period=0.0, history_length=1)
    task_settings: dict = {}


def build_env_cfg(settings, device="cuda:0", seed=1):
    validate_settings(settings)
    spec = robot_spec(settings.asset.name)
    names = spec.joints
    urdf = prepare_urdf(robot=spec.name)
    limits = joint_contract(robot=spec.name)
    motor = DCMotorLimiter(names, torch.tensor([limits[n]["effort"] for n in names]),
                          torch.tensor([limits[n]["velocity"] for n in names]), settings.mujica.motor)
    a, p = settings.asset, settings.sim.physx
    cfg = MUJICAEnvCfg(
        seed=seed, task_settings=json.loads(json.dumps(settings)), state_space=spec.critic_dim,
        decimation=settings.control.decimation, episode_length_s=settings.env.episode_length_s,
        sim=sim_utils.SimulationCfg(device=device, dt=settings.sim.dt,
            render_interval=settings.control.decimation, gravity=tuple(settings.sim.gravity),
            physx=sim_utils.PhysxCfg(solver_type=p.solver_type,
                bounce_threshold_velocity=p.bounce_threshold_velocity,
                gpu_max_rigid_contact_count=p.max_gpu_contact_pairs)),
        scene=InteractiveSceneCfg(num_envs=settings.env.num_envs, env_spacing=settings.env.env_spacing,
                                  replicate_physics=True),
        robot=ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot",
            spawn=sim_utils.UrdfFileCfg(
                asset_path=str(urdf), usd_dir=str(urdf.parent / "usd"), usd_file_name=f"{spec.name}.usd",
                fix_base=a.fix_base_link, merge_fixed_joints=False, activate_contact_sensors=True,
                self_collision=(a.self_collisions == 0),
                replace_cylinders_with_capsules=False,
                joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                    target_type="none", gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=a.disable_gravity, linear_damping=a.linear_damping,
                    angular_damping=a.angular_damping, max_linear_velocity=a.max_linear_velocity,
                    max_angular_velocity=a.max_angular_velocity, max_depenetration_velocity=p.max_depenetration_velocity),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=p.contact_offset, rest_offset=p.rest_offset),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=(a.self_collisions == 0),
                    solver_position_iteration_count=p.num_position_iterations,
                    solver_velocity_iteration_count=p.num_velocity_iterations)),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=tuple(settings.init_state.pos),
                # The source settings store xyzw; Lab uses wxyz.
                rot=tuple([settings.init_state.rot[3]] + settings.init_state.rot[:3]),
                lin_vel=tuple(settings.init_state.lin_vel), ang_vel=tuple(settings.init_state.ang_vel),
                joint_pos=dict(settings.init_state.default_joint_angles), joint_vel={".*": 0.0}),
            actuators={"effort": ImplicitActuatorCfg(
                joint_names_expr=[".*"], stiffness=0.0, damping=0.0,
                effort_limit_sim={name: float(motor.peak[i]) for i, name in enumerate(names)},
                # X5 uses URDF speed caps; Go2W applies its torque envelope in Python.
                velocity_limit_sim=({n: limits[n]['velocity'] for n in names} if spec.name == 'x5' else 1.0e9),
                armature=a.armature, friction=0.0)},
            soft_joint_pos_limit_factor=settings.rewards.soft_dof_pos_limit),
    )
    if spec.name == 'x5':
        # Decorate once here: the wrapper applies overrides before cloning.
        from isaaclab.sim.utils import clone
        cfg.robot.spawn.func = clone(spawn_x5_from_urdf)
        cfg.contact_sensor.history_length = 3
    cfg.viewer.eye = (10.0, 0.0, 6.0)
    cfg.viewer.lookat = (4.0, 4.0, 0.0)
    return cfg
