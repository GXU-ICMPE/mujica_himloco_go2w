"""Simulator-independent behavior for the three terrain locomotion tasks.

Reward adaptation is shared with the retained Gym backend.
"""
import torch

from .skills import SKILL_NAMES, TERRAIN_GROUPS
from .rewards import TerrainRewards


class LocomotionTasks(TerrainRewards):
    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        for index, key in enumerate(("lin_vel_x", "lin_vel_y", "ang_vel_yaw")):
            lo, hi = self.command_ranges[key]
            self.commands[env_ids, index] = torch.rand(len(env_ids), device=self.device)*(hi-lo)+lo
        self.commands[env_ids, 3] = 0.0
        self.commands[env_ids, :2] *= (self.commands[env_ids, :2].norm(dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions):
        if self.settings.control.control_type != "P":
            raise ValueError("MUJICA uses leg position targets and wheel velocity targets")
        q_error = self.default_dof_pos - self.dof_pos
        q_error[:, self.wheel_indices] = 0.0
        position_target = actions * self.settings.control.action_scale
        position_target[:, self.wheel_indices] = 0.0
        velocity_target = torch.zeros_like(actions)
        velocity_target[:, self.wheel_indices] = actions[:, self.wheel_indices] * self.settings.control.vel_scale
        requested = (self.p_gains*self.Kp_factors*(position_target+q_error) +
                     self.d_gains*self.Kd_factors*(velocity_target-self.dof_vel))
        requested = requested * self.motor_strength_factors
        self.motor_violation = self.motor_limiter.violation_count(requested, self.dof_pos, self.dof_vel).float()
        # All three locomotion skills are driven from the first physics step.
        return self.motor_limiter.clip(requested, self.dof_pos, self.dof_vel)

    def check_termination(self):
        # Neither the assigned terrain skill nor the S2 selected skill changes
        # failure handling or the episode clock.
        self.failure_buf = (self.contact_forces[:, self.termination_contact_indices].norm(dim=-1) > 1.0).any(dim=1)
        self.terrain_exit_buf = torch.zeros_like(self.failure_buf)
        if self.custom_origins:
            half_extent = torch.tensor([self.settings.terrain.terrain_length, self.settings.terrain.terrain_width],
                                       device=self.device)*0.5 - self.settings.mujica.terrain_edge_margin
            offset = (self.root_states[:, :2] - self.env_origins[:, :2]).abs()
            self.terrain_exit_buf = (offset >= half_extent).any(dim=1)
        limit = round(self.settings.env.episode_length_s / self.dt)
        # Exiting a tile truncates the rollout before another task's geometry
        # can be collected with this tile's S1 skill label.
        self.time_out_buf = ((self.episode_length_buf >= limit) | self.terrain_exit_buf) & ~self.failure_buf
        self.reset_buf = self.failure_buf | self.time_out_buf

    def _update_progress_metrics(self):
        self.command_distance += self.commands[:, :2].norm(dim=1) * self.dt
        tracking = self._reward_tracking_lin_vel() >= self.settings.mujica.tracking_success_fraction
        self.tracking_streak = torch.where(tracking, self.tracking_streak+self.dt, 0.0)
        self.tracking_best_streak = torch.maximum(self.tracking_best_streak, self.tracking_streak)

    def _update_terrain_curriculum(self, env_ids):
        if not self.init_done or not self.custom_origins:
            return
        ids = env_ids[self.episode_length_buf[env_ids] > 0]
        if len(ids) == 0:
            return
        distance = (self.root_states[ids, :2] - self.episode_start_xy[ids]).norm(dim=1)
        up = ((self.tracking_best_streak[ids] >= self.settings.mujica.tracking_success_seconds)
              & (self.command_distance[ids] > 0.5)
              & (distance >= self.settings.mujica.curriculum_min_distance)
              & ~self.failure_buf[ids])
        down = self.failure_buf[ids] | (distance < self.settings.mujica.distance_failure_fraction*self.command_distance[ids])
        levels = self.terrain_levels[ids] + up.long() - (down & ~up).long()
        # Pure flat ground has no difficulty parameter to increase.
        is_flat = torch.tensor([name == "flat" for name in self.terrain.terrain_names], device=self.device)
        self.terrain_levels[ids] = torch.where(is_flat[self.terrain_types[ids]], 0, levels.clamp(0, self.max_terrain_level-1))
        self.env_origins[ids] = self.terrain_origins[self.terrain_levels[ids], self.terrain_types[ids]]

    def _episode_skill_metrics(self, env_ids):
        """Report physical task outcomes, separately from S2 selection fractions."""
        completed = env_ids[self.episode_length_buf[env_ids] > 0]
        metrics = {}
        groups = [("task/"+name, completed[self.task_ids[completed] == index])
                  for index, name in enumerate(SKILL_NAMES)]
        if self.custom_origins:
            for name in (kind for group in TERRAIN_GROUPS for kind in group):
                columns = torch.tensor([kind == name for kind in self.terrain.terrain_names], device=self.device)
                groups.append(("terrain/"+name, completed[columns[self.terrain_types[completed]]]))
        for prefix, ids in groups:
            if len(ids) == 0:
                continue
            steps = self.episode_length_buf[ids].float()
            metrics[prefix+"/episodes"] = torch.tensor(float(len(ids)), device=self.device)
            for key, term in (("tracking_lin", "tracking_lin_vel"), ("tracking_yaw", "tracking_ang_vel")):
                metrics[prefix+"/"+key] = (self.episode_sums[term][ids]/(steps*self.reward_scales[term])).mean()
            metrics[prefix+"/distance"] = (self.root_states[ids, :2]-self.episode_start_xy[ids]).norm(dim=1).mean()
            metrics[prefix+"/failure_fraction"] = self.failure_buf[ids].float().mean()
            metrics[prefix+"/tile_exit_fraction"] = self.terrain_exit_buf[ids].float().mean()
            if prefix.startswith("task/") and hasattr(self, "_reward_unclipped_sum"):
                names = self.reward_scales if self.stage == "s1" else ("tracking_lin_vel", "tracking_ang_vel")
                for name in names:
                    metrics[prefix+"/reward/"+name] = (self.episode_sums[name][ids]/(steps*self.dt)).mean()
                metrics[prefix+"/reward_unclipped"] = (self._reward_unclipped_sum[ids]/(steps*self.dt)).mean()
                metrics[prefix+"/reward_clipped_fraction"] = (self._reward_clipped_steps[ids]/steps).mean()
        return metrics
