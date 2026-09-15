"""Offline reward behavior, geometry reference and reset/metadata regressions."""
import copy

import pytest
import torch

from mujica.config import training_config
from mujica.isaaclab.settings import default_settings, validate_settings
from mujica.rewards import POSTURE_TERMS, TerrainRewards, reward_metadata
from mujica.runner import MUJICARunner
from mujica.smoke import MockVecEnv, smoke_config
from mujica.train import parser, prepare_training
from test_environment_contract import MODULE
from test_isaaclab_migration import NativeEnv, tensor_env


def reward_env(stage="s1"):
    env = tensor_env(stage)
    env.task_ids[:] = torch.arange(3)
    env.skill_ids[:] = torch.tensor([2, 0, 1])
    env.dof_pos[:] = env.default_dof_pos
    for buffer in (env.dof_vel, env.last_dof_vel, env.actions, env.last_actions,
                   env.torques, env.contact_forces, env.base_ang_vel, env.measured_heights):
        buffer.zero_()
    env.commands[:, :3] = torch.tensor([0.5, 0., 0.])
    env.base_lin_vel[:] = env.commands[:, :3]
    env.projected_gravity[:] = torch.tensor([0., 0., -1.])
    env.root_states[:, 2] = 0.4
    for value in env.episode_sums.values():
        value.zero_()
    return env


def bend_legs(env, amount=0.3):
    for leg in range(4):
        env.dof_pos[:, 4*leg+1:4*leg+3] += amount


def test_both_backends_dispatch_to_the_shared_reward_implementation():
    for cls in (NativeEnv, MODULE.MUJICARobot):
        for name in ("compute_reward", "_reward_base_height", "_reward_run_still", "_reward_stand_still"):
            # The Gym import shim restores sys.modules after loading, so the
            # same source module may have two Python object identities.
            actual, expected = getattr(cls, name), getattr(TerrainRewards, name)
            assert actual.__module__ == expected.__module__
            assert actual.__qualname__ == expected.__qualname__
            assert actual.__code__.co_filename == expected.__code__.co_filename


def test_effective_weights_follow_physical_task_and_preserve_other_terms():
    env = reward_env()
    expected = {"orientation": [-0.5, -0.35, -0.35], "base_height": [-10., -7., -7.],
                "run_still": [-0.05, -0.025, -0.025], "stand_still": [-0.5, -0.35, -0.35]}
    for name, weights in expected.items():
        torch.testing.assert_close(env._reward_weight(name), torch.tensor(weights)*env.dt)
    for name in env.reward_scales:
        if name not in POSTURE_TERMS:
            assert env._reward_weight(name) == env.settings.rewards.scales[name]*env.dt
    bend_legs(env)
    env.compute_reward()
    torch.testing.assert_close(env.rew_buf, torch.tensor([2.25-0.12, 2.25-0.06, 2.25-0.06])*env.dt)
    before = env.rew_buf.clone()
    env.set_skill(torch.tensor([0, 1, 2]))
    env.compute_reward()
    torch.testing.assert_close(env.rew_buf, before)


def test_pure_turn_is_active_and_stationary_legs_have_complex_terrain_tolerance():
    env = reward_env()
    bend_legs(env)
    env.dof_pos[:, env.wheel_indices] = 99.
    original = env.dof_pos.clone()
    env.commands[:, :3] = torch.tensor([0., 0., 0.5])
    assert env._reward_stand_still().count_nonzero() == 0
    torch.testing.assert_close(env._reward_run_still(), torch.full((3,), 2.4))
    env.commands.zero_()
    assert env._reward_run_still().count_nonzero() == 0
    torch.testing.assert_close(env._reward_stand_still(), torch.tensor([2.4, 1.6, 1.6]))
    torch.testing.assert_close(env.dof_pos, original)
    env.dof_pos[:] = env.default_dof_pos
    bend_legs(env, amount=0.05)
    torch.testing.assert_close(env._reward_stand_still(), torch.tensor([0.4, 0., 0.]))
    for command in ([0.1, 0., 0.], [0., 0., 0.1]):
        env.commands[:, :3] = torch.tensor(command)
        assert not env._stationary_command().any()  # no unpenalized threshold gap


def test_height_uses_local_support_region_without_changing_observation_samples():
    env = reward_env()
    # 17x11 scan. The central 9x7 patch spans x +/-0.4m and y +/-0.3m.
    env.measured_heights.fill_(0.2)
    env.measured_heights.view(3, 17, 11)[:, 4:13, 2:9] = 0.
    before = env.measured_heights.clone()
    torch.testing.assert_close(env._reward_height_reference(), torch.tensor([0.2*124/187, 0., 0.]))
    assert env._reward_height_indices.numel() == 63
    assert env._reward_base_height()[0] > 0
    assert env._reward_base_height()[1:].count_nonzero() == 0
    torch.testing.assert_close(env.measured_heights, before)
    env.measured_heights.view(3, 17, 11)[:, 4:13, 2:9] = 0.1
    torch.testing.assert_close(env._reward_base_height()[1:], torch.full((2,), 0.07**2))


def test_height_tolerance_remains_bounded_and_invariant_to_absolute_elevation():
    env = reward_env()
    env.root_states[:, 2] = torch.tensor([0.42, 0.42, 0.38])
    torch.testing.assert_close(env._reward_base_height(), torch.tensor([0.02**2, 0., 0.]))
    env.root_states[:, 2] = torch.tensor([0.48, 0.48, 0.32])
    expected = torch.tensor([0.08**2, 0.05**2, 0.05**2])
    torch.testing.assert_close(env._reward_base_height(), expected)
    env.root_states[:, 2] += 1.7
    env.measured_heights += 1.7
    torch.testing.assert_close(env._reward_base_height(), expected)


@pytest.mark.parametrize("positive_clip", [True, False])
def test_reward_diagnostics_use_actual_weights_and_reset_only_completed_rows(positive_clip):
    env = reward_env()
    env.settings.rewards.only_positive_rewards = positive_clip
    env.torques[0] = 100.  # 16*100^2*-0.0005 = -80, beyond the tracking reward
    env.episode_length_buf[:] = 1
    env.compute_reward()
    assert env.rew_buf[0] == pytest.approx(0. if positive_clip else (2.25-80)*env.dt)
    metrics = env._episode_skill_metrics(torch.arange(3))
    assert metrics['task/flat_slope/reward/torques'].item() == pytest.approx(-80.)
    assert metrics['task/flat_slope/reward_unclipped'].item() == pytest.approx(2.25-80.)
    assert metrics['task/flat_slope/reward_clipped_fraction'].item() == float(positive_clip)
    assert metrics['task/discrete/reward_clipped_fraction'].item() == 0.
    remaining = env._reward_unclipped_sum[1:].clone()
    env._reset_idx(torch.tensor([0]))
    assert env._reward_unclipped_sum[0] == 0
    assert env._reward_clipped_steps[0] == 0
    torch.testing.assert_close(env._reward_unclipped_sum[1:], remaining)
    assert env.extras['episode']['task/flat_slope/reward_unclipped'].item() == pytest.approx(2.25-80.)


def test_s2_rewards_remain_tracking_only_regardless_of_posture_or_selected_skill():
    env = reward_env("s2")
    env.torques.fill_(100.)
    bend_legs(env, amount=1.)
    env.root_states[:, 2] = 0.9
    env.compute_reward()
    torch.testing.assert_close(env.rew_buf, torch.full((3,), 2.25*env.dt))
    for name, values in env.episode_sums.items():
        if name not in ("tracking_lin_vel", "tracking_ang_vel"):
            assert values.count_nonzero() == 0
    assert env._reward_clipped_steps.count_nonzero() == 0


def test_checkpoint_snapshot_restores_reward_parameters_without_default_substitution(tmp_path):
    env = reward_env()
    env.settings.rewards.terrain_adaptation.complex_scales['orientation'] = -0.4
    metadata = env.export_metadata()
    assert metadata['reward_profile']['parameter_source'] == 'task_ids'
    assert metadata['reward_profile']['s1_weights']['discrete']['orientation'] == -0.4
    metadata['environment_config'] = copy.deepcopy(dict(env.settings))
    saved = {'config': training_config('s1'), 'metadata': metadata}
    checkpoint = tmp_path/'s1.pt'
    torch.save(saved, checkpoint)
    _, settings = prepare_training(parser().parse_args(['--resume', str(checkpoint), '--check-config']))
    assert settings.rewards.terrain_adaptation.complex_scales['orientation'] == -0.4
    del saved['metadata']['environment_config']['rewards']['terrain_adaptation']
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match='start a new S1'):
        prepare_training(parser().parse_args(['--resume', str(checkpoint), '--check-config']))


def test_direct_runner_resume_rejects_a_changed_reward_profile(tmp_path):
    runner = MUJICARunner(MockVecEnv(), smoke_config(), tmp_path/'run')
    checkpoint = tmp_path/'old.pt'
    runner.save(checkpoint)
    runner.metadata['reward_profile'] = reward_metadata(default_settings().rewards)
    with pytest.raises(ValueError, match='Resume reward profile differs'):
        runner.load(checkpoint)
    if runner.writer:
        runner.writer.close()


@pytest.mark.parametrize('field,value', [('complex_base_height_tolerance', -0.01),
                                       ('stand_still_yaw_threshold', 0.), ('version', 99)])
def test_invalid_reward_adaptation_fails_before_simulator_start(field, value):
    settings = default_settings()
    settings.rewards.terrain_adaptation[field] = value
    with pytest.raises(ValueError):
        validate_settings(settings)
