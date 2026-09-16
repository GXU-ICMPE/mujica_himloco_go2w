"""X5loco v8 posture/landing state adapted to MUJICA tensors and heightfields.

Copyright 2026 Robot-Nav. Apache-2.0; see third_party/X5loco-LICENSE.
Modified: removed manager/scanner dependencies; explicit policy joint order.
"""
import math
import torch
from .math import quat_apply, quat_rotate_inverse as quat_apply_inverse


def smooth_gate(value, limits):
    t = ((value - limits[0]) / (limits[1] - limits[0])).clamp(0, 1)
    return t.square() * (3 - 2 * t)


class X5V8TerrainState:
    def __init__(self, env):
        self.env, self.cfg = env, env.settings.rewards.terrain_geometry
        xy = env.height_points[0, :, :2].cpu()
        design = torch.cat((xy, torch.ones_like(xy[:, :1])), -1)
        self.design = design.to(env.device)
        self.inverse = torch.linalg.pinv(design).to(env.device)
        pairs = ((torch.cdist(xy, xy) - 0.1).abs() < 1e-4).triu(1).nonzero()
        self.pair_a, self.pair_b = pairs.to(env.device).unbind(-1)
        self.flat = torch.zeros(env.num_envs, device=env.device)
        self.tilt_cost = torch.zeros_like(self.flat)
        self.last_step = -1

    def update(self):
        env, cfg = self.env, self.cfg
        if self.last_step == env.common_step_counter:
            return self
        self.last_step = env.common_step_counter
        hits = env.measured_heights
        valid = torch.isfinite(hits).all(-1) & (hits.abs().amax(-1) < 1e6)
        heights = torch.where(valid[:, None], hits, 0.)
        heights = heights - heights.mean(-1, keepdim=True)
        plane = heights @ self.inverse.T
        residual = heights - plane @ self.design.T
        slope_deg = torch.rad2deg(torch.atan(plane[:, :2].norm(dim=-1)))
        rough = torch.maximum(smooth_gate(residual.square().mean(-1).sqrt(), cfg.residual_transition_m),
            smooth_gate((residual[:, self.pair_a] - residual[:, self.pair_b]).abs().amax(-1), cfg.jump_transition_m))
        slope = (1 - rough) * smooth_gate(slope_deg, cfg.slope_transition_deg)
        flat = torch.where(valid, 1 - rough - slope, 0.)
        rough = torch.where(valid, rough, 0.)
        slope = torch.where(valid, slope, 1.)
        allowance = (flat * cfg.flat_tilt_allowance_deg
            + slope * (slope_deg + cfg.slope_tilt_margin_deg).clamp(max=cfg.slope_tilt_cap_deg)
            + rough * cfg.rough_tilt_allowance_deg)
        allowance = torch.where(valid, allowance, cfg.invalid_tilt_allowance_deg)
        gravity = env.projected_gravity
        tilt = torch.atan2(gravity[:, :2].norm(dim=-1), -gravity[:, 2])
        self.flat = flat
        self.tilt_cost = (flat + (1-flat)*cfg.nonflat_tilt_scale) * (tilt - torch.deg2rad(allowance)).clamp_min(0).square()
        return self

    def reset(self, env_ids):
        self.flat[env_ids] = 0
        self.tilt_cost[env_ids] = 0


class X5V8MotionState:
    def __init__(self, env):
        self.env, self.cfg = env, env.settings.rewards.posture
        self.reference_xy = torch.tensor(self.cfg.reference_wheel_xy_m, device=env.device)
        self.reference_pose = self.reference_xy.new_tensor(self.cfg.stand_joint_angles)
        self.maneuver = torch.zeros(env.num_envs, device=env.device)
        self.stop_time = torch.zeros_like(self.maneuver)
        self.contact_time = torch.zeros(env.num_envs, 4, device=env.device)
        self.was_supported = torch.zeros_like(self.contact_time, dtype=torch.bool)
        self.air_started = torch.zeros_like(self.was_supported)
        self.air_time = torch.zeros_like(self.contact_time)
        self.peak_clearance = torch.zeros_like(self.contact_time)
        self.air_progress = torch.zeros_like(self.contact_time)
        self.last_step = -1
        self.costs = {name: torch.zeros_like(self.maneuver) for name in (
            "joint_pose", "wheel_position", "wheelbase", "track_width", "height", "maneuver_step",
        )}
        self.samples = torch.zeros_like(self.maneuver)
        self.sums = {name: torch.zeros_like(self.maneuver) for name in (
            "height_m", "height_error_m", "xy_error_rms_m", "front_retraction_m",
            "wheelbase_m", "track_width_m", "maneuver_gate", "stop_gate", "steps_per_second",
        )}

    def _tensor(self, value):
        return self.reference_xy.new_tensor(value)

    def update(self):
        env, cfg = self.env, self.cfg
        if self.last_step == env.common_step_counter:
            return self
        self.last_step = env.common_step_counter
        flat = env.x5_terrain.update().flat
        nonflat = 1 - flat
        commands = env.commands[:, :3]
        terrain_cfg = env.settings.rewards.terrain_geometry
        requested_maneuver = torch.maximum(
            smooth_gate(commands[:, 1].abs(), terrain_cfg.lateral_transition),
            smooth_gate(commands[:, 2].abs(), terrain_cfg.yaw_transition),
        )
        # Open the corridor immediately; close it gradually after a turn.
        self.maneuver = torch.maximum(requested_maneuver,
                                      self.maneuver * math.exp(-env.step_dt / cfg.recenter_time_s))
        stopping = ((commands[:, :2].norm(dim=-1) < terrain_cfg.stop_linear_threshold)
                    & (commands[:, 2].abs() < terrain_cfg.stop_yaw_threshold))
        # Command-based tightening also acts on a robot that is still drifting.
        self.stop_time = torch.where(stopping, self.stop_time + env.step_dt, 0.0)
        stop = smooth_gate(self.stop_time, (0.0, cfg.stop_tighten_time_s))
        maneuver = self.maneuver
        pose_band = (self._tensor(cfg.rolling_joint_deadband)
                     + maneuver[:, None] * (self._tensor(cfg.maneuver_joint_deadband)
                                             - self._tensor(cfg.rolling_joint_deadband)))
        pose_band = ((1 - stop[:, None]) * pose_band + stop[:, None] * self._tensor(cfg.stop_joint_deadband)
                     + nonflat[:, None] * self._tensor(cfg.nonflat_extra_joint_deadband))
        q = env.dof_pos[:, env.leg_indices].reshape(-1, 4, 3)
        pose_error = ((q - self.reference_pose).abs() - pose_band[:, None, :]).clamp_min(0)
        self.costs["joint_pose"] = (flat + 0.15 * nonflat) * (
            pose_error / self._tensor(cfg.joint_error_scale)).square().mean(dim=(1, 2))

        centers_w = env.feet_pos
        offset_w = centers_w - env.root_states[:, None, :3]
        quat = env.base_quat[:, None, :].expand(-1, 4, -1)
        xy = quat_apply_inverse(quat.reshape(-1, 4), offset_w.reshape(-1, 3)).reshape(-1, 4, 3)[..., :2]
        error_xy = xy - self.reference_xy
        xy_band = (self._tensor(cfg.rolling_xy_deadband_m)
                   + maneuver[:, None] * self._tensor(cfg.maneuver_extra_xy_deadband_m))
        xy_band = ((1 - stop[:, None]) * xy_band + stop[:, None] * self._tensor(cfg.stop_xy_deadband_m)
                   + nonflat[:, None] * self._tensor(cfg.nonflat_extra_xy_deadband_m))
        # A finite penalty survives turning and lateral commands. No contact
        # gate: lifting a wheel cannot switch off its body-frame constraints.
        geometry_scale = flat * (1 - 0.6 * maneuver * (1 - stop)) + 0.12 * nonflat
        self.costs["wheel_position"] = geometry_scale * (
            (error_xy.abs() - xy_band[:, None, :]).clamp_min(0) / cfg.xy_error_scale_m
        ).square().sum(-1).mean(-1)
        wheelbase = xy[:, :2, 0] - xy[:, 2:, 0]
        width = xy[:, [1, 3], 1] - xy[:, [0, 2], 1]
        ref_base = self.reference_xy[:2, 0] - self.reference_xy[2:, 0]
        ref_width = self.reference_xy[[1, 3], 1] - self.reference_xy[[0, 2], 1]
        base_allow = cfg.wheelbase_shortening_m + 0.10 * maneuver * (1 - stop) + 0.15 * nonflat
        width_allow = cfg.track_narrowing_m + 0.08 * maneuver * (1 - stop) + 0.12 * nonflat
        span_scale = flat + 0.15 * nonflat
        self.costs["wheelbase"] = span_scale * (
            (ref_base - base_allow[:, None] - wheelbase).clamp_min(0) / cfg.span_error_scale_m
        ).square().mean(-1)
        self.costs["track_width"] = span_scale * (
            (ref_width - width_allow[:, None] - width).clamp_min(0) / cfg.span_error_scale_m
        ).square().mean(-1)

        hits = env.x5_small_heights
        valid = torch.isfinite(hits).all(-1) & (hits.abs().amax(-1) < 1.0e6)
        height = env.root_states[:, 2] - torch.nan_to_num(hits).mean(-1)
        height = torch.where(valid, height, cfg.reference_base_height_m)
        height_band = ((1 - stop) * (cfg.rolling_height_deadband_m
                                    + maneuver * cfg.maneuver_extra_height_deadband_m)
                       + stop * cfg.stop_height_deadband_m + nonflat * cfg.nonflat_extra_height_deadband_m)
        self.costs["height"] = (flat + 0.15 * nonflat) * (
            ((height - cfg.reference_base_height_m).abs() - height_band).clamp_min(0) / cfg.height_error_scale_m
        ).square()
        step_reward, step_rate = self._steps(centers_w, commands, requested_maneuver)
        self.costs["maneuver_step"] = step_reward
        values = (height, (height - cfg.reference_base_height_m).abs(),
                  error_xy.square().sum(-1).mean(-1).sqrt(), (-error_xy[:, :2, 0]).clamp_min(0).mean(-1),
                  wheelbase.mean(-1), width.mean(-1), maneuver, stop, step_rate)
        for total, value in zip(self.sums.values(), values):
            total += value
        self.samples += 1
        return self

    def _steps(self, centers_w, commands, maneuver):
        """One payout after an actual lift/landing, coupled to signed base motion."""
        env, cfg = self.env, self.cfg
        forces = env.contact_forces[:, env.feet_indices]
        contact = forces.norm(dim=-1) > cfg.contact_threshold_n
        self.contact_time = torch.where(contact, self.contact_time + env.step_dt, 0.0)
        supported = ((self.contact_time >= cfg.contact_confirmation_s)
                     & (forces[..., 2] > cfg.support_vertical_force_n))
        liftoff = self.was_supported & ~contact
        self.air_started |= liftoff
        self.air_time = torch.where(self.air_started, self.air_time + env.step_dt, 0.0)

        # Reuse the existing rays; query axle-adjacent terrain, not a global
        # minimum which mistakes a stair edge for large swing clearance.
        hits = env.x5_scan_hits
        valid_hits = torch.isfinite(hits).all(-1) & (hits.abs().amax(-1) < 1.0e6)
        safe_hits = torch.where(valid_hits[..., None], hits, 0.0)
        dist2 = (centers_w[:, :, None, :2] - safe_hits[:, None, :, :2]).square().sum(-1)
        dist2 = torch.where(valid_hits[:, None, :], dist2, float("inf"))
        nearest_dist2, nearest = dist2.min(-1)
        ground = torch.gather(safe_hits[..., 2], 1, nearest)
        wheel_q = env.rigid_body_states[:, env.feet_indices, 3:7]
        axes = self._tensor((0.0, 1.0, 0.0)).expand(env.num_envs * 4, -1)
        axis_z = quat_apply(wheel_q.reshape(-1, 4), axes)[:, 2].reshape(-1, 4)
        rim_extent = env.settings.mujica.wheel_radius * (1 - axis_z.square()).clamp_min(0).sqrt()
        clearance = (centers_w[..., 2] - ground - rim_extent).clamp_min(0)
        valid = nearest_dist2 <= cfg.scan_nearest_radius_m ** 2
        clearance = torch.where(valid & self.air_started & ~contact, clearance, 0.0)
        self.peak_clearance = torch.maximum(self.peak_clearance, clearance)
        # Integrate signed progress so back-and-forth rocking does not earn the
        # same credit as net motion. Yaw is scaled to an equivalent rim speed.
        desired = torch.stack((commands[:, 1], 0.5 * commands[:, 2]), dim=-1)
        actual = torch.stack((env.base_lin_vel[:, 1],
                              0.5 * env.base_ang_vel[:, 2]), dim=-1)
        progress = ((desired * actual).sum(-1) / desired.square().sum(-1).clamp_min(0.01)).clamp(-1, 1)
        tracking = torch.exp(-((actual - desired) / 0.3).square().sum(-1))
        self.air_progress += self.air_started * (progress * maneuver * env.step_dt)[:, None]
        landed = self.air_started & supported
        valid_step = (landed & (self.air_time >= cfg.step_air_time_range_s[0])
                      & (self.air_time <= cfg.step_air_time_range_s[1])
                      & (self.peak_clearance >= cfg.step_min_clearance_m)
                      & (supported.sum(-1) >= 2)[:, None])
        productivity = (self.air_progress / self.air_time.clamp_min(env.step_dt)).clamp(0, 1)
        height_score = (self.peak_clearance / cfg.step_target_clearance_m).clamp(0, 1)
        # RewardManager multiplies by dt: dividing here makes each confirmed
        # landing an event reward independent of the policy update frequency.
        reward = (valid_step * productivity * height_score).mean(-1) * tracking * maneuver / env.step_dt
        step_rate = valid_step.float().sum(-1) / env.step_dt
        clear = landed | (self.air_time > cfg.step_air_time_range_s[1])
        for value in (self.air_time, self.peak_clearance, self.air_progress):
            value[clear] = 0
        self.air_started[clear] = False
        self.was_supported[:] = supported
        return reward, step_rate

    def reset(self, env_ids):
        count = self.samples[env_ids].clamp_min(1)
        log = {f"X5V8/motion/{name}": (value[env_ids] / count).mean() for name, value in self.sums.items()}
        for value in (*self.costs.values(), *self.sums.values(), self.samples, self.maneuver, self.stop_time,
                      self.contact_time, self.was_supported, self.air_started, self.air_time,
                      self.peak_clearance, self.air_progress):
            value[env_ids] = 0
        return log
