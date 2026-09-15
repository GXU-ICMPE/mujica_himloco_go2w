"""Offline regression checks. Physics bindings are excluded, never simulated.

Exercise the real task methods against shuffled joint state and known tensors,
including DirectRLEnv's reward-before-auto-reset ordering.
"""
import ast
import math
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from mujica.isaaclab.assets import prepare_urdf, transform, inertial_properties
from mujica.isaaclab.settings import JOINT_NAMES, URDF_PATH, default_settings, joint_contract, joint_indices, validate_checkpoint_backend
from mujica.isaaclab.math import quat_apply, quat_rotate_inverse, quat_from_euler_xyz, yaw_rotate
from mujica.isaaclab.task_logic import COLLISION_GROUPS
from mujica.isaaclab.terrain import HeightField
from mujica.isaaclab.adapter import MUJICAVecEnv
from mujica.train import parser, prepare_training
from mujica.skills import skill_metadata, TERRAIN_GROUPS

ROOT = Path(__file__).resolve().parents[1]


class OfflineDirectEnv:
    """Only the counter reset supplied by the base class; no physics stand-in."""
    def _reset_idx(self, ids):
        self.episode_length_buf[ids] = 0


def load_task_class():
    path = ROOT / "mujica/isaaclab/env.py"
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if not (
        isinstance(n, ast.ImportFrom) and (n.module or "").startswith("isaaclab")) and not (
        isinstance(n, ast.Import) and any(a.name.startswith("isaaclab") for a in n.names))]
    namespace = {"__name__": "mujica.isaaclab._offline_contract", "__package__": "mujica.isaaclab",
                 "DirectRLEnv": OfflineDirectEnv}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["MUJICAEnv"]


NativeEnv = load_task_class()


def tensor_env(stage="s1"):
    env = NativeEnv.__new__(NativeEnv)
    env.settings = default_settings()
    env.settings.terrain.mesh_type = "plane"
    env.settings.mujica.stage = stage
    env.settings.noise.add_noise = False
    env.settings.domain_rand.push_robots = False
    env.num_envs, env.num_actions, env.num_dof, env.num_bodies = 3, 16, 16, 19
    env.device, env.stage, env.dt, env.physics_dt = "cpu", stage, 0.02, 0.005
    env.urdf_path = URDF_PATH
    env.dof_names = list(JOINT_NAMES)
    # Simulate the common breadth-first ordering; compare against policy order.
    sim_names = [f"{leg}_{segment}_joint" for segment in ("hip", "thigh", "calf", "foot")
                 for leg in ("RR", "FL", "RL", "FR")]
    env.joint_ids = torch.tensor(joint_indices(sim_names))
    env.body_names = [n for group in COLLISION_GROUPS for n in group]
    env.sensor_ids = torch.arange(19)
    env.collision_body_groups = [torch.tensor([env.body_names.index(n) for n in group]) for group in COLLISION_GROUPS]
    env.clearance_body_indices = torch.tensor([env.body_names.index(f"{leg}_foot") for leg in ("FL", "FR", "RL", "RR")])
    env.feet_indices = env.clearance_body_indices
    env.wheel_indices, env.hip_indices = torch.tensor([3, 7, 11, 15]), torch.tensor([0, 4, 8, 12])
    env.base_index = 0
    env.termination_contact_indices = torch.tensor([0])
    env.penalised_contact_indices = torch.tensor([i for i, n in enumerate(env.body_names) if any(p in n for p in ("base", "thigh", "calf"))])
    root_state = torch.zeros(3, 13)
    root_state[:, 2], root_state[:, 3] = 0.45, 1.0
    q = torch.tensor([[env.settings.init_state.default_joint_angles[n] for n in sim_names]]).repeat(3, 1)
    body = torch.zeros(3, 19, 13)
    body[..., 3] = 1
    body[:, env.clearance_body_indices, 2] = 0.086
    env.robot = SimpleNamespace(joint_names=sim_names, data=SimpleNamespace(
        root_state_w=root_state.clone(), default_root_state=root_state.clone(),
        joint_pos=q.clone(), default_joint_pos=q.clone(), joint_vel=torch.zeros_like(q), body_state_w=body))
    env.robot.write_root_state_to_sim = lambda state, ids: env.robot.data.root_state_w.__setitem__(ids, state)
    def write_joints(pos, vel, env_ids):
        env.robot.data.joint_pos[env_ids] = pos
        env.robot.data.joint_vel[env_ids] = vel
    env.robot.write_joint_state_to_sim = write_joints
    env.contact_sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w=torch.zeros(3, 19, 3)))
    env.scene = SimpleNamespace(env_origins=torch.tensor([[0., 0., 0.], [4., 0., 0.], [8., 0., 0.]]))
    env.cfg = SimpleNamespace(decimation=4, robot=SimpleNamespace(spawn=SimpleNamespace(asset_path="prepared.urdf")))
    env.custom_origins = False
    env._initialize_buffers()
    env.episode_length_buf = torch.zeros(3, dtype=torch.long)
    env.common_step_counter = 0
    env.init_done = True
    env.extras = {}
    env._randomize_reset_properties = lambda ids: None
    env._reset_idx(torch.arange(3))
    env._get_observations()
    return env


def system_inertia(root):
    # Evaluate all links at q=0. Compare total mass, COM and inertia about world
    # origin, catching lost rotations and missing parallel-axis contributions.
    poses = {"base": np.eye(4)}
    pending = list(root.findall("joint"))
    while pending:
        for joint in pending[:]:
            parent = joint.find("parent").get("link")
            if parent in poses:
                poses[joint.find("child").get("link")] = poses[parent] @ transform(joint.find("origin"))
                pending.remove(joint)
    total, moment, inertia = 0.0, np.zeros(3), np.zeros((3, 3))
    for link in root.findall("link"):
        mass, center, local = inertial_properties(link)
        pose = poses[link.get("name")]
        center = pose[:3, :3] @ center + pose[:3, 3]
        inertia += pose[:3, :3] @ local @ pose[:3, :3].T
        inertia += mass * (np.dot(center, center)*np.eye(3) - np.outer(center, center))
        total += mass
        moment += mass * center
    return total, moment/total, inertia


def test_fixed_link_conversion_preserves_mass_com_inertia_and_collisions(tmp_path):
    original = ET.parse(URDF_PATH).getroot()
    prepared = ET.parse(prepare_urdf(cache_dir=tmp_path)).getroot()
    assert len(prepared.findall("link")) == 19
    assert len(prepared.findall(".//collision")) == len(original.findall(".//collision"))
    assert {j.get("name") for j in prepared.findall("joint") if j.get("type") != "fixed"} == set(JOINT_NAMES)
    for before, after in zip(system_inertia(original), system_inertia(prepared)):
        np.testing.assert_allclose(before, after, atol=1e-10)
    for name in JOINT_NAMES:
        a, b = original.find(f"joint[@name='{name}']"), prepared.find(f"joint[@name='{name}']")
        assert a.find("axis").attrib == b.find("axis").attrib
        assert a.find("limit").attrib == b.find("limit").attrib


def test_asset_meshes_and_policy_joint_mapping():
    assert len(joint_contract()) == 16
    order = list(reversed(JOINT_NAMES))
    assert [order[i] for i in joint_indices(order)] == list(JOINT_NAMES)
    with pytest.raises(ValueError):
        joint_indices(order[:-1])


def test_settings_match_complete_current_gym_inheritance_chain():
    namespace = {"BaseConfig": object}
    for file in ("legged_gym/envs/base/legged_robot_config.py",
                 "legged_gym/envs/go2w/go2w_config.py", "legged_gym/envs/mujica/mujica_config.py"):
        tree = ast.parse((ROOT/file).read_text())
        tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        exec(compile(tree, file, "exec"), namespace)
    def values(cls):
        return {name: values(getattr(cls, name)) if isinstance(getattr(cls, name), type) else getattr(cls, name)
                for name in dir(cls) if not name.startswith("_")}
    original = values(namespace["MUJICACfg"])
    original.pop("viewer")
    assert default_settings() == original


def test_wxyz_rotation_matches_independent_scipy_and_broadcasts():
    angles = torch.tensor([[0.4, -0.2, 1.1], [0., math.pi, 0.]], dtype=torch.float64)
    q = quat_from_euler_xyz(*angles.unbind(-1))
    v = torch.tensor([[0.5, 1., -0.4], [1., 0., -1.]], dtype=torch.float64)
    rotated = quat_apply(q, v)
    np.testing.assert_allclose(rotated, Rotation.from_euler("xyz", angles.numpy()).apply(v.numpy()), atol=1e-12)
    torch.testing.assert_close(quat_rotate_inverse(q, rotated), v)
    assert quat_apply(q[:, None], v[:, None].expand(-1, 187, -1)).shape == (2, 187, 3)
    assert yaw_rotate(q, torch.zeros(1, 187, 3)).shape == (2, 187, 3)


def test_three_skill_terrain_variants_and_curriculum_levels():
    settings = default_settings()
    settings.terrain.num_cols = 18
    terrain = HeightField(settings.terrain)
    assert set(terrain.terrain_names) == {"flat", "stairs_up", "stairs_down", "slope_up", "slope_down", "discrete"}
    for name, task in zip(terrain.terrain_names, terrain.task_ids):
        assert name in TERRAIN_GROUPS[task]
    assert np.all(terrain.env_origins[..., 2] == 0)
    assert np.isfinite(terrain.height_field_raw).all()
    assert terrain.env_origins.shape == (20, 18, 3)


def test_frames_preserve_wheels_and_contact_group_semantics():
    env = tensor_env()
    env.contact_forces[0, env.body_names.index("Head_lower"), 2] = 5
    env.dof_pos[:, env.wheel_indices] = 27
    before_q, before_qd = env.dof_pos.clone(), env.dof_vel.clone()
    frame, privileged = env._current_frame()
    assert frame.shape == (3, 58) and privileged.shape == (3, 270)
    assert privileged[0, 62] == 1 and privileged[0, 61:79].sum() == 1
    assert frame[:, 9:25][:, env.wheel_indices].count_nonzero() == 0
    env._reward_dof_vel()
    torch.testing.assert_close(env.dof_pos, before_q)
    torch.testing.assert_close(env.dof_vel, before_qd)
    assert torch.isfinite(privileged).all()


@pytest.mark.parametrize("stage", ["s1", "s2"])
def test_partial_auto_reset_keeps_terminal_successor_and_one_history_step(stage):
    env = tensor_env(stage)
    env.add_noise = True
    old = env.history_buf.clone()
    env.episode_length_buf[:] = torch.tensor([1000, 2, 300])
    env.check_termination()
    done = env.reset_buf.clone()
    reward = env._get_rewards()
    terminal = env.extras["terminal_critic"].clone()
    successors = env.extras["next_privileged_observations"].clone()
    env._reset_idx(done.nonzero().flatten())
    obs = env._get_observations()
    assert obs["policy"].shape == (3, 348) and reward.shape == (3,)
    torch.testing.assert_close(env.extras["terminal_critic"], terminal)
    torch.testing.assert_close(terminal, successors[done])
    torch.testing.assert_close(obs["policy"][~done, 58:], old[~done, :-58])
    history = obs["policy"][done].reshape(-1, 6, 58)
    torch.testing.assert_close(history[:, 0:1].expand_as(history), history)
    # Native reset writes joint positions in simulator order, preserving ABI.
    torch.testing.assert_close(env.robot.data.joint_pos[done][:, env.joint_ids], env.dof_pos[done])
    same = env._get_observations()["policy"]
    torch.testing.assert_close(same, obs["policy"])


def test_selector_switch_does_not_rewrite_past_or_change_task():
    env = tensor_env("s2")
    history, tasks, episodes = env.history_buf.clone(), env.task_ids.clone(), env.episode_length_buf.clone()
    selector_obs = env.selector_obs.clone()
    env.set_skill(torch.tensor([2, 0, 1]))
    torch.testing.assert_close(env.history_buf[:, 58:], history[:, 58:])
    torch.testing.assert_close(env.task_ids, tasks)
    torch.testing.assert_close(env.episode_length_buf, episodes)
    torch.testing.assert_close(env.selector_obs, selector_obs)
    assert env.history_buf[:, 57].tolist() == [2., 0., 1.]


def test_all_skills_active_from_start_and_terminal_precedence():
    env = tensor_env("s1")
    env.task_ids[:] = torch.arange(3)
    env.settings.mujica.motor.enabled = False
    env.episode_length_buf[:] = 0
    effort = env._compute_torques(torch.ones(3, 16))
    assert (effort.abs().sum(dim=1) > 0).all()
    env.episode_length_buf[:] = 100
    assert env._compute_torques(torch.ones(3, 16))[2].count_nonzero() > 0
    env.episode_length_buf[:] = torch.tensor([1000, 1000, 300])
    env.contact_forces[:, 0, 2] = 4
    env.check_termination()
    assert env.reset_buf.tolist() == [True, True, True]
    assert env.time_out_buf.tolist() == [False, False, False]
    env.contact_forces.zero_()
    env.check_termination()
    assert env.reset_buf.tolist() == [True, True, False]


def test_explicit_mixed_pd_is_scattered_by_joint_names():
    env = tensor_env()
    env.settings.domain_rand.delay = False
    env.settings.domain_rand.disturbance = False
    env.episode_length_buf[:] = 101
    captured = {}
    env.robot.set_joint_effort_target = lambda effort, joint_ids: captured.update(effort=effort.clone(), indices=joint_ids)
    env.robot.permanent_wrench_composer = SimpleNamespace(set_forces_and_torques=lambda **kwargs: None)
    env._pre_physics_step(torch.full((3, 16), 0.1))
    env._apply_action()
    assert captured["indices"].tolist() == joint_indices(env.robot.joint_names)
    torch.testing.assert_close(captured["effort"][:, env.wheel_indices], torch.full((3, 4), 0.5))
    assert torch.isfinite(captured["effort"]).all()


def test_rewards_match_bundled_gym_formulas_for_same_tensors():
    env = tensor_env()
    source = ROOT / "legged_gym/envs/base/legged_robot.py"
    cls = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef))
    rewards = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name.startswith("_reward_")]
    scope = {"torch": torch}
    exec(compile(ast.Module(body=rewards, type_ignores=[]), str(source), "exec"), scope)
    torch.manual_seed(7)
    env.cfg = env.settings  # only for calling unmodified upstream reward methods
    env.actions.normal_()
    env.dof_vel.normal_()
    env.contact_forces.normal_()
    # Baseline task, active translation: unchanged formulas match upstream.
    # Revised stationary/yaw and complex-terrain semantics have separate tests.
    env.commands[:, :3] = torch.tensor([0.5, 0.0, 0.0])
    for name, weight in env.settings.rewards.scales.items():
        if name == "dof_vel":  # MUJICA fixes upstream's in-place wheel corruption
            continue
        expected = scope["_reward_"+name](env)
        actual = getattr(env, "_reward_"+name)()
        torch.testing.assert_close(actual, expected)
        assert env.reward_scales[name] == pytest.approx(weight * 0.02)
    env.compute_reward()
    expected_total = sum(getattr(env, "_reward_"+n)()*w for n, w in env.reward_scales.items()).clamp(min=0)
    torch.testing.assert_close(env.rew_buf, expected_total)
    env.stage = "s2"
    env.compute_reward()
    expected_s2 = env._reward_tracking_lin_vel()*0.03 + env._reward_tracking_ang_vel()*0.015
    torch.testing.assert_close(env.rew_buf, expected_s2)


def test_checkpoint_simulator_and_joint_order_are_required_for_resume():
    saved = {"metadata": {**skill_metadata(), "simulator": "isaaclab", "joint_names": list(JOINT_NAMES)}}
    validate_checkpoint_backend(saved, resume=True)
    saved["metadata"]["simulator"] = "isaacgym"
    with pytest.raises(ValueError, match="Isaac Lab checkpoint"):
        validate_checkpoint_backend(saved, resume=True)
    validate_checkpoint_backend(saved, resume=False)
    saved["metadata"]["joint_names"].reverse()
    with pytest.raises(ValueError, match="joint order"):
        validate_checkpoint_backend(saved, resume=False)


def test_cli_default_backend_and_early_validation():
    args = parser().parse_args(["--check-config"])
    assert args.backend == "isaaclab" and args.num_envs == 64
    config, settings = prepare_training(args)
    assert config["stage"] == "s1" and settings.env.num_envs == 64
    with pytest.raises(ValueError, match="requires --low-level"):
        prepare_training(parser().parse_args(["--stage", "s2"]))
    with pytest.raises(ValueError, match="Terrain requires"):
        prepare_training(parser().parse_args(["--terrain-rows", "0"]))


def test_adapter_exposes_real_terminal_rows_and_stable_policy_metadata():
    env = tensor_env("s2")
    env.episode_length_buf[:] = torch.tensor([1000, 10, 10])
    env.check_termination()
    rewards = env._get_rewards()
    info = env.extras.copy()
    terminal = info["terminal_critic"].clone()
    terminated, timeouts = env.reset_buf & ~env.time_out_buf, env.time_out_buf.clone()
    env._reset_idx(torch.tensor([0]))
    obs = env._get_observations()
    env.step = lambda actions: (obs, rewards, terminated, timeouts, info)
    adapter = MUJICAVecEnv(env)
    policy, critic, _, dones, result, ids, saved_terminal = adapter.step(torch.zeros(3, 16))
    assert policy.shape == (3, 348) and critic.shape == (3, 270)
    assert dones.tolist() == [True, False, False] and ids.tolist() == [0]
    torch.testing.assert_close(saved_terminal, terminal)
    metadata = adapter.export_metadata()
    assert metadata["simulator"] == "isaaclab"
    assert metadata["joint_names"] == list(JOINT_NAMES)
    assert metadata["control_dt"] == 0.02
    assert metadata["simulator_joint_names"] != metadata["joint_names"]
