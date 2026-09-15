"""Shared terrain locomotion rewards for Isaac Lab and the retained Gym backend.

The terrain task selects posture parameters. The policy-selected skill never
changes the scoring rule. S2 keeps its two velocity-tracking terms.
"""
import math
import torch

from .skills import SKILL_NAMES

POSTURE_TERMS = ("orientation", "base_height", "run_still", "stand_still")


def _values(obj):
    return obj if isinstance(obj, dict) else {name: getattr(obj, name) for name in dir(obj) if not name.startswith("_")}


def validate_reward_settings(rewards, terrain):
    cfg = _values(rewards)
    if "terrain_adaptation" not in cfg:
        raise ValueError("Reward settings predate shared terrain posture v1; start a new S1 run with the current defaults")
    profile = _values(cfg["terrain_adaptation"])
    if profile.get("version") != 1:
        raise ValueError("Unsupported terrain reward adaptation version")
    overrides = profile["complex_scales"]
    scales = _values(cfg["scales"])
    if set(overrides) != set(POSTURE_TERMS):
        raise ValueError("Complex-terrain overrides must contain exactly the four posture penalties")
    for name, value in overrides.items():
        if not math.isfinite(value) or not scales[name] <= value < 0:
            raise ValueError("Complex-terrain posture weights must remain negative and no stronger than the baseline")
    for name in ("stand_still_linear_threshold", "stand_still_yaw_threshold", "base_height_patch_half_length",
                 "base_height_patch_half_width"):
        if not math.isfinite(profile[name]) or profile[name] <= 0:
            raise ValueError(name + " must be finite and positive")
    for name in ("complex_stand_still_joint_tolerance", "complex_base_height_tolerance"):
        if not math.isfinite(profile[name]) or profile[name] < 0:
            raise ValueError(name + " must be finite and non-negative")
    terrain = _values(terrain)
    xs, ys = terrain["measured_points_x"], terrain["measured_points_y"]
    if not any(abs(x) <= profile["base_height_patch_half_length"] for x in xs) or not any(
            abs(y) <= profile["base_height_patch_half_width"] for y in ys):
        raise ValueError("Height reward patch must contain measured terrain points")


def reward_metadata(rewards):
    cfg = _values(rewards)
    profile = dict(_values(cfg["terrain_adaptation"]))
    scales = dict(_values(cfg["scales"]))
    return dict(name="shared_terrain_posture_v1", parameter_source="task_ids",
                parameters=profile,
                s1_weights={name: dict(scales, **(profile["complex_scales"] if i else {}))
                            for i, name in enumerate(SKILL_NAMES)},
                s2_weights={name: scales[name] for name in ("tracking_lin_vel", "tracking_ang_vel")})


class TerrainRewards:
    def _reward_weight(self, name):
        """Effective per-environment weight, including dt exactly once."""
        overrides = self.settings.rewards.terrain_adaptation.complex_scales
        if self.stage == "s1" and name in overrides:
            return torch.where(self.task_ids != 0, overrides[name] * self.dt, self.reward_scales[name])
        return self.reward_scales[name]

    def _stationary_command(self):
        cfg = self.settings.rewards.terrain_adaptation
        return ((self.commands[:, :2].norm(dim=1) < cfg.stand_still_linear_threshold)
                & (self.commands[:, 2].abs() < cfg.stand_still_yaw_threshold))

    def _reward_stand_still(self):
        cfg = self.settings.rewards.terrain_adaptation
        error = (self.dof_pos - self.default_dof_pos).abs()
        tolerance = torch.where(self.task_ids != 0, cfg.complex_stand_still_joint_tolerance, 0.0)
        error = (error - tolerance[:, None]).clamp(min=0)
        error[:, self.wheel_indices] = 0
        return error.sum(dim=1) * self._stationary_command()

    def _reward_run_still(self):
        error = (self.dof_pos - self.default_dof_pos).abs()
        error[:, self.wheel_indices] = 0
        # Pure yaw is an active motion command, not a request to park.
        return error.sum(dim=1) * ~self._stationary_command()

    def _reward_height_reference(self):
        if not hasattr(self, "_reward_height_indices"):
            cfg = self.settings.rewards.terrain_adaptation
            terrain = self.settings.terrain
            # Same x-major flattening as the 17x11 observation height grid.
            indices = [i * len(terrain.measured_points_y) + j
                       for i, x in enumerate(terrain.measured_points_x)
                       for j, y in enumerate(terrain.measured_points_y)
                       if abs(x) <= cfg.base_height_patch_half_length + 1e-6
                       and abs(y) <= cfg.base_height_patch_half_width + 1e-6]
            self._reward_height_indices = torch.tensor(indices, dtype=torch.long, device=self.device)
        local = self.measured_heights[:, self._reward_height_indices].mean(dim=1)
        return torch.where(self.task_ids != 0, local, self.measured_heights.mean(dim=1))

    def _reward_base_height(self):
        error = (self.root_states[:, 2] - self._reward_height_reference()
                 - self.settings.rewards.base_height_target).abs()
        tolerance = torch.where(self.task_ids != 0,
            self.settings.rewards.terrain_adaptation.complex_base_height_tolerance, 0.0)
        return (error - tolerance).clamp(min=0).square()

    def compute_reward(self):
        self.rew_buf.zero_()
        names = self.reward_scales if self.stage == "s1" else ("tracking_lin_vel", "tracking_ang_vel")
        for name in names:
            term = getattr(self, "_reward_" + name)() * self._reward_weight(name)
            self.rew_buf += term
            self.episode_sums[name] += term
        if not hasattr(self, "_reward_unclipped_sum"):
            self._reward_unclipped_sum = torch.zeros_like(self.rew_buf)
            self._reward_clipped_steps = torch.zeros_like(self.rew_buf)
        self._reward_unclipped_sum += self.rew_buf
        if self.stage == "s1" and self.settings.rewards.only_positive_rewards:
            self._reward_clipped_steps += (self.rew_buf < 0).float()
            self.rew_buf.clamp_(min=0)

    def _reset_reward_diagnostics(self, env_ids):
        if hasattr(self, "_reward_unclipped_sum"):
            self._reward_unclipped_sum[env_ids] = 0
            self._reward_clipped_steps[env_ids] = 0
