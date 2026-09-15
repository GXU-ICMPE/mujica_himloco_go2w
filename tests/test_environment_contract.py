"""CPU-only environment contract tests. These do not simulate Isaac/PhysX.

Import shims cover unavailable Isaac bindings; observation/termination/reset
methods under test are the production environment methods, not reimplementations.
"""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_environment_module():
    fake_gym = types.ModuleType("isaacgym")
    fake_gym.gymapi = types.SimpleNamespace()
    fake_gym.gymtorch = types.SimpleNamespace(unwrap_tensor=lambda x: x)
    fake_utils = types.ModuleType("isaacgym.torch_utils")
    fake_utils.quat_from_euler_xyz = lambda *x: None
    fake_utils.quat_rotate_inverse = lambda quat, xyz: xyz.clone()
    fake_utils.torch_rand_float = lambda lo, hi, shape, device: torch.rand(shape, device=device)*(hi-lo)+lo
    fake_base = types.ModuleType("legged_gym.envs.go2w.go2w_robot")
    fake_base.Go2w = object
    fake_package = types.ModuleType("_mujica_environment_contract")
    fake_package.__path__ = []
    fake_terrain = types.ModuleType("_mujica_environment_contract.mujica_terrain")
    fake_terrain.MUJICATerrain = object
    injected = {"isaacgym": fake_gym, "isaacgym.torch_utils": fake_utils,
                "legged_gym.envs.go2w.go2w_robot": fake_base,
                "_mujica_environment_contract": fake_package,
                "_mujica_environment_contract.mujica_terrain": fake_terrain}
    with patch.dict(sys.modules, injected):
        spec = importlib.util.spec_from_file_location(
            "_mujica_environment_contract.mujica_robot",
            ROOT / "legged_gym/envs/mujica/mujica_robot.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


MODULE = load_environment_module()
S = types.SimpleNamespace


def bare_env(stage="s2"):
    env = MODULE.MUJICARobot.__new__(MODULE.MUJICARobot)
    env.num_envs, env.num_bodies, env.device, env.stage = 2, 19, "cpu", stage
    env.dt = 1.0
    env.cfg = S(normalization=S(clip_observations=100.0),
                terrain=S(mesh_type="plane"), env=S(episode_length_s=20.0),
                mujica=S(contact_threshold=1.0, wheel_radius=0.086))
    env.settings = env.cfg
    env.custom_origins = False
    env.failure_buf = torch.zeros(2, dtype=torch.bool)
    env.terrain_exit_buf = torch.zeros(2, dtype=torch.bool)
    env.episode_start_xy = torch.zeros(2, 2)
    env.obs_scales = S(ang_vel=0.25, lin_vel=2.0, dof_pos=1.0,
                       dof_vel=0.05, height_measurements=5.0)
    env.commands_scale = torch.tensor([2.0, 2.0, 0.25])
    env.dof_pos = torch.arange(32).view(2, 16).float()/10
    env.default_dof_pos = torch.zeros(1, 16)
    env.dof_vel = torch.ones(2, 16)*2.0
    env.wheel_indices = torch.tensor([3, 7, 11, 15])
    env.skill_ids = torch.tensor([0, 2])
    env.task_ids = torch.tensor([0, 2])
    env.actions = torch.ones(2, 16)
    env.commands = torch.zeros(2, 4)
    env.base_ang_vel = torch.zeros(2, 3)
    env.base_lin_vel = torch.tensor([[1., 0., 0.], [2., 0., 0.]])
    env.projected_gravity = torch.tensor([[0., 0., -1.]]).repeat(2, 1)
    env.root_states = torch.zeros(2, 13)
    env.root_states[:, 2] = 0.4
    env.root_states[:, 6] = 1.0
    env.root_states[:, 7:10] = env.base_lin_vel
    env.base_quat = env.root_states[:, 3:7]
    env.gravity_vec = env.projected_gravity.clone()
    env.rigid_body_states = torch.zeros(2*19, 13)
    env.rigid_body_states[:, 2] = 0.086
    env.contact_forces = torch.zeros(2, 19, 3)
    env.collision_body_groups = [torch.tensor([0]), torch.tensor([1, 2])] + [torch.tensor([i]) for i in range(3, 19)]
    env.clearance_body_indices = torch.tensor([6, 10, 14, 18])
    env.feet_indices = env.clearance_body_indices
    env.termination_contact_indices = torch.tensor([0])
    env.measured_heights = torch.zeros(2, 187)
    env.add_noise = False
    env.obs_buf = torch.zeros(2, 348)
    env.privileged_obs_buf = torch.zeros(2, 270)
    env._pending_reset = torch.ones(2, dtype=torch.bool)
    env.episode_length_buf = torch.zeros(2, dtype=torch.long)
    return env


class EnvironmentContractTests(unittest.TestCase):
    def test_frame_channels_collision_groups_and_no_live_joint_mutation(self):
        env = bare_env()
        before_q, before_qd = env.dof_pos.clone(), env.dof_vel.clone()
        env.contact_forces[0, 2, 2] = 3.0  # lower head contributes to real grouped target
        env.rigid_body_states.view(2, 19, 13)[1, env.clearance_body_indices, 2] += 0.2
        obs, priv = env._current_frame()
        self.assertEqual(tuple(obs.shape), (2, 58))
        self.assertEqual(tuple(priv.shape), (2, 270))
        self.assertTrue(torch.equal(priv[:, 58:61], env.base_lin_vel*2.0))
        self.assertEqual(priv[0, 62].item(), 1.0)
        self.assertEqual(priv[0, 61:79].sum().item(), 1.0)
        self.assertTrue(torch.allclose(priv[1, 79:83], torch.full((4,), 0.2)))
        self.assertTrue(torch.equal(env.dof_pos, before_q))
        self.assertTrue(torch.equal(env.dof_vel, before_qd))
        expected = before_qd.clone()
        expected[:, env.wheel_indices] = 0.0
        self.assertTrue(torch.equal(env._reward_dof_vel(), expected.square().sum(dim=1)))
        self.assertTrue(torch.equal(env.dof_vel, before_qd))

    def test_selector_skill_changes_only_latest_frame_and_no_task_state(self):
        env = bare_env()
        env.compute_observations()
        old_history = env.obs_buf.clone()
        old_selector = env.selector_obs.clone()
        old_task = env.task_ids.clone()
        env.set_skill(torch.tensor([1, 0]))
        self.assertTrue(torch.equal(env.obs_buf[:, 58:], old_history[:, 58:]))
        self.assertTrue(torch.equal(env.selector_obs, old_selector))
        self.assertTrue(torch.equal(env.task_ids, old_task))
        self.assertTrue(torch.equal(env.obs_buf[:, 57], torch.tensor([1., 0.])))
        self.assertTrue(torch.equal(env.privileged_obs_buf[:, 57], torch.tensor([1., 0.])))
        self.assertEqual(env.episode_length_buf.sum().item(), 0)

    def test_uniform_duration_and_selection_independent_failure_handling(self):
        env = bare_env("s1")
        env.contact_forces[:, 0, 2] = 10.0
        env.episode_length_buf[:] = 5
        env.check_termination()
        self.assertEqual(env.reset_buf.tolist(), [True, True])
        env.episode_length_buf[1] = 6
        env.check_termination()
        self.assertEqual(env.time_out_buf.tolist(), [False, False])
        env.episode_length_buf[0] = 20
        env.check_termination()
        self.assertTrue(env.reset_buf[0].item())
        self.assertFalse(env.time_out_buf[0].item())
        env.episode_length_buf[0] = 5
        env.stage = "s2"
        env.check_termination()
        self.assertEqual(env.reset_buf.tolist(), [True, True])
        env.skill_ids[:] = torch.tensor([2, 0])
        env.check_termination()
        self.assertEqual(env.reset_buf.tolist(), [True, True])
        env.contact_forces.zero_()
        env.episode_length_buf[:] = 20
        env.check_termination()
        self.assertTrue(bool(env.time_out_buf.all()))

    def test_task_metrics_use_completed_episode_state(self):
        env = bare_env("s1")
        env.episode_length_buf[:] = torch.tensor([10, 6])
        env.episode_sums = {"tracking_lin_vel": torch.tensor([7.5, 4.5]), "tracking_ang_vel": torch.tensor([3.75, 2.25])}
        env.reward_scales = {"tracking_lin_vel": 1.5, "tracking_ang_vel": 0.75}
        env.failure_buf[1] = True
        metrics = env._episode_skill_metrics(torch.tensor([0, 1]))
        self.assertEqual(metrics["task/flat_slope/tracking_lin"].item(), 0.5)
        self.assertEqual(metrics["task/stairs/failure_fraction"].item(), 1.0)
        env.task_ids[0] = 1
        env.root_states[0, :2] = torch.tensor([1.5, 0.0])
        metrics = env._episode_skill_metrics(torch.tensor([0]))
        self.assertEqual(metrics["task/discrete/distance"].item(), 1.5)
        env.episode_length_buf.zero_()
        self.assertEqual(env._episode_skill_metrics(torch.tensor([0, 1])), {})

    def test_successor_saved_before_reset_and_history_advances_once(self):
        env = bare_env()
        env.compute_observations()
        previous = env.obs_buf.clone()
        env.actions[0] = 3.0
        env.actions[1] = 5.0
        env.episode_length_buf[:] = torch.tensor([19, 0])
        env.common_step_counter = 0
        env.gym = S(refresh_actor_root_state_tensor=lambda x: None,
                    refresh_net_contact_force_tensor=lambda x: None,
                    refresh_rigid_body_state_tensor=lambda x: None)
        env.sim = object()
        env._post_physics_step_callback = lambda: None
        env._update_progress_metrics = lambda: None
        env.compute_reward = lambda: None
        env.motor_violation = torch.zeros(2)
        env.last_actions = torch.zeros(2, 16)
        env.last_last_actions = torch.zeros(2, 16)
        env.last_dof_vel = torch.zeros(2, 16)
        env.last_root_vel = torch.zeros(2, 6)
        env.disturbance = torch.zeros(2, 19, 3)
        env.viewer = None

        def reset_selected(ids):
            env.actions[ids] = 0.0
            env.base_lin_vel[ids] = 0.0
            env.episode_length_buf[ids] = 0
            env._pending_reset[ids] = True
        env.reset_idx = reset_selected
        ids, terminal = env.post_physics_step()
        self.assertEqual(ids.tolist(), [0])
        self.assertEqual(terminal[0, 58].item(), 2.0)
        self.assertEqual(env.privileged_obs_buf[0, 58].item(), 0.0)
        self.assertEqual(terminal[0, 41].item(), 3.0)
        self.assertEqual(env.obs_buf[0, 41].item(), 0.0)
        self.assertTrue(torch.equal(env.obs_buf[1, 58:], previous[1, :-58]))
        self.assertTrue(torch.equal(env.obs_buf[0].view(6, 58), env.obs_buf[0, :58].repeat(6, 1)))
        self.assertEqual(env.extras["next_observations"][0, 41].item(), 3.0)
        self.assertTrue(env.extras["time_outs"][0].item())


if __name__ == "__main__":
    unittest.main()
