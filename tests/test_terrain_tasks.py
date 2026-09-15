"""Task semantics, balanced terrain coverage and checkpoint compatibility."""

import numpy as np
import pytest
import torch

from mujica.skills import SKILL_NAMES, SKILL_VALUES, TERRAIN_GROUPS
from mujica.terrain import HeightField, assign_terrain_columns
from mujica.isaaclab.settings import default_settings
from mujica.export import export_checkpoint
from mujica.runner import MUJICARunner
from mujica.smoke import MockVecEnv, smoke_config
from mujica.sim2sim import validate_metadata
from test_isaaclab_migration import tensor_env
from test_sim2sim_mapping import metadata as deployment_metadata


@pytest.mark.parametrize("count", [3, 9, 64, 512])
def test_task_allocation_is_balanced_despite_unequal_variant_counts(count):
    cfg = default_settings().terrain
    cfg.num_rows = 1
    terrain = HeightField(cfg)
    columns = assign_terrain_columns(count, terrain.task_ids)
    tasks = terrain.task_ids[columns]
    counts = np.bincount(tasks, minlength=3)
    assert counts.max()-counts.min() <= 1
    if count >= 9:
        assert {terrain.terrain_names[c] for c in columns} == {name for group in TERRAIN_GROUPS for name in group}


def test_slope_and_stair_direction_is_defined_outwards_from_spawn():
    cfg = default_settings().terrain
    cfg.num_rows, cfg.num_cols = 20, 6
    terrain = HeightField(cfg)
    cx, cy = terrain.nx//2, terrain.ny//2
    for name in ("slope_up", "slope_down", "stairs_up", "stairs_down"):
        col = terrain.terrain_names.index(name)
        easy, _ = terrain.tile(0, col)
        hard, _ = terrain.tile(19, col)
        assert np.all(hard[cx-3:cx+4, cy-3:cy+4] == 0)
        sign = 1 if name.endswith("up") else -1
        for x, y in ((cx+25, cy), (cx-25, cy), (cx, cy+25), (cx, cy-25)):
            assert sign*hard[x, y] > sign*easy[x, y] > 0
        outward = sign*hard[cx:cx+30, cy].astype(float)
        assert (np.diff(outward) >= 0).all()
    up, _ = terrain.tile(19, terrain.terrain_names.index("stairs_up"))
    steps = up[cx:cx+30, cy]
    edges = np.flatnonzero(np.diff(steps) > 0)
    assert np.all(np.diff(edges) == round(cfg.stair_width/cfg.horizontal_scale))


def test_flat_is_flat_and_discrete_obstacles_are_irregular():
    cfg = default_settings().terrain
    cfg.num_rows, cfg.num_cols = 1, 6
    np.random.seed(9)
    terrain = HeightField(cfg)
    flat, _ = terrain.tile(19, terrain.terrain_names.index("flat"))
    discrete, _ = terrain.tile(19, terrain.terrain_names.index("discrete"))
    assert np.count_nonzero(flat) == 0
    assert len(np.unique(discrete[discrete > 0])) > 5
    assert discrete.max()*cfg.vertical_scale <= cfg.discrete_height_range[1]+cfg.vertical_scale
    assert np.all(discrete[terrain.nx//2-3:terrain.nx//2+4, terrain.ny//2-3:terrain.ny//2+4] == 0)


@pytest.mark.parametrize("stage", ["s1", "s2"])
def test_all_tasks_reset_upright_and_receive_velocity_commands(stage):
    env = tensor_env(stage)
    env.task_ids[:] = torch.arange(3)
    env.skill_ids[:] = torch.tensor([2, 0, 1])
    env.command_ranges = {"lin_vel_x": [0.4, 0.4], "lin_vel_y": [0.3, 0.3], "ang_vel_yaw": [0.2, 0.2]}
    env._reset_idx(torch.arange(3))
    torch.testing.assert_close(env.root_states[:, 3:7], torch.tensor([[1., 0., 0., 0.]]).repeat(3, 1))
    torch.testing.assert_close(env.root_states[:, 2]-env.env_origins[:, 2], torch.full((3,), 0.45))
    torch.testing.assert_close(env.commands[:, :3], torch.tensor([[0.4, 0.3, 0.2]]).repeat(3, 1))
    assert (env._compute_torques(torch.ones(3, 16)).abs().sum(dim=1) > 0).all()
    if stage == "s1":
        assert env.skill_ids.tolist() == [0, 1, 2]
    env.episode_length_buf[:] = 300
    env.check_termination()
    assert not env.reset_buf.any()  # no old six-second Recovery timeout
    env.episode_length_buf[:] = 1000
    env.check_termination()
    assert env.time_out_buf.all()


@pytest.mark.parametrize("stage", ["s1", "s2"])
def test_leaving_tile_truncates_and_collision_takes_precedence(stage):
    env = tensor_env(stage)
    env.custom_origins = True
    env.root_states[:, :2] = env.env_origins[:, :2]
    env.root_states[:2, 0] += 3.6  # beyond 8m tile minus 0.5m edge margin
    env.contact_forces[1, 0, 2] = 2
    env.check_termination()
    assert env.terrain_exit_buf.tolist() == [True, True, False]
    assert env.reset_buf.tolist() == [True, True, False]
    assert env.time_out_buf.tolist() == [True, False, False]


def test_curriculum_requires_travel_and_preserves_task_columns():
    env = tensor_env()
    cfg = default_settings().terrain
    cfg.num_cols = 6
    env.terrain = HeightField(cfg)
    env.terrain_origins = torch.tensor(env.terrain.env_origins)
    env.terrain_types = torch.tensor([0, 3, 4])  # flat, discrete, stairs
    env.terrain_levels = torch.ones(3, dtype=torch.long)
    env.max_terrain_level = cfg.num_rows
    env.custom_origins = True
    env.task_ids = torch.arange(3)
    env.episode_length_buf[:] = 200
    env.episode_start_xy = env.root_states[:, :2].clone()
    env.root_states[:, 0] += 2
    env.command_distance[:] = 3
    env.tracking_best_streak[:] = 4
    env.failure_buf.zero_()
    env._update_terrain_curriculum(torch.arange(3))
    assert env.terrain_levels.tolist() == [0, 2, 2]
    assert env.terrain_types.tolist() == [0, 3, 4]
    env.root_states[:, :2] = env.episode_start_xy
    env._update_terrain_curriculum(torch.arange(3))
    assert env.terrain_levels.tolist() == [0, 1, 1]  # standing on the spawn patch cannot promote
    env.root_states[:, 0] += 2
    env.failure_buf[:] = True
    env._update_terrain_curriculum(torch.arange(3))
    assert env.terrain_levels.tolist() == [0, 0, 0]


def test_plane_diagnostics_only_assign_flat_slope():
    env = tensor_env()
    assert env.task_ids.tolist() == [0, 0, 0]
    assert env.skill_ids.tolist() == [0, 0, 0]


def test_old_skill_checkpoints_fail_for_resume_s2_export_and_sim2sim(tmp_path):
    runner = MUJICARunner(MockVecEnv(), smoke_config(), tmp_path/"s1")
    path = tmp_path/"new.pt"
    runner.save(path)
    saved = torch.load(path, weights_only=False)
    saved["metadata"].update(skill_names=["moving", "climb", "recovery"], task_contract_version=1)
    old = tmp_path/"old.pt"
    torch.save(saved, old)
    with pytest.raises(ValueError, match="Incompatible skill contract"):
        runner.load(old)
    with pytest.raises(ValueError, match="Incompatible skill contract"):
        MUJICARunner(MockVecEnv(), smoke_config("s2"), tmp_path/"s2", low_level=old)
    with pytest.raises(ValueError, match="Incompatible skill contract"):
        export_checkpoint(old, tmp_path/"old_export.pt")
    assert not (tmp_path/"old_export.pt").exists()
    meta = deployment_metadata()
    meta["skill_values"] = {"moving": 0, "climb": 1, "recovery": 2}
    with pytest.raises(ValueError, match="Incompatible skill contract"):
        validate_metadata(meta)
    if runner.writer:
        runner.writer.close()


def test_cli_accepts_only_final_skill_names():
    from mujica.train import parser
    from mujica.train_isaacgym import parser as gym_parser
    from mujica.sim2sim import SKILLS
    assert SKILL_NAMES == ("flat_slope", "discrete", "stairs")
    assert SKILLS == {"auto": -1, **SKILL_VALUES}
    for make_parser in (parser, gym_parser):
        for name in SKILL_NAMES:
            assert make_parser().parse_args(["--skill", name]).skill == name
        with pytest.raises(SystemExit):
            make_parser().parse_args(["--skill", "recovery"])
