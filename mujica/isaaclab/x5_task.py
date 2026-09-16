"""X5loco v8 control and reward adapter for MUJICA's DirectRLEnv.

Copyright 2026 Robot-Nav. Apache-2.0; see third_party/X5loco-LICENSE.
Modified: tensor interface, terrain-only curriculum, S1/S2 reward dispatch.
"""
import torch
from .math import torch_rand_float, yaw_rotate
from .x5_state import X5V8TerrainState, X5V8MotionState, smooth_gate


class X5Task:
    def _initialize_x5(self):
        self.training_iteration = 0
        self.x5_terrain = X5V8TerrainState(self)
        self.x5_motion = X5V8MotionState(self)
        self.command_time_left = torch.zeros(self.num_envs, device=self.device)
        self.wheel_policy_target = torch.zeros(self.num_envs, 4, device=self.device)
        self.stop_integral = torch.zeros_like(self.wheel_policy_target)
        self.stop_contact_time = torch.zeros_like(self.wheel_policy_target)
        self.stop_control_time = torch.zeros(self.num_envs, device=self.device)
        self.stop_requested = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.motor_offset = torch.zeros_like(self.actions)
        limits = self.settings.control.position_target_limits
        self.leg_target_min = torch.tensor([limits[n][0] for n in self.dof_names[:12]], device=self.device)
        self.leg_target_max = torch.tensor([limits[n][1] for n in self.dof_names[:12]], device=self.device)
        # Match v8's 0.4 x 0.3 m small scanner, including its half-grid Y offsets.
        xy = torch.stack(torch.meshgrid(torch.linspace(-.2, .2, 5, device=self.device),
            torch.linspace(-.15, .15, 4, device=self.device), indexing='ij'), -1).reshape(1, -1, 2)
        self.x5_small_points = torch.cat((xy, torch.zeros_like(xy[..., :1])), -1)

    def _update_x5_scans(self):
        xyz = yaw_rotate(self.base_quat, self.height_points) + self.root_states[:, None, :3]
        self.x5_scan_hits = torch.cat((xyz[..., :2], self.measured_heights[..., None]), -1)
        small = yaw_rotate(self.base_quat, self.x5_small_points) + self.root_states[:, None, :3]
        self.x5_small_heights = self._sample_ground_at(small)

    def _resample_commands(self, env_ids):
        if not self.is_x5:
            return super()._resample_commands(env_ids)
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        ranges = self.command_ranges
        for i, key in enumerate(('lin_vel_x', 'lin_vel_y', 'ang_vel_yaw')):
            self.commands[env_ids, i] = torch_rand_float(*ranges[key], (n,), self.device)
        self.commands[env_ids, 3] = 0.
        self.command_time_left[env_ids] = torch_rand_float(8., 10., (n,), self.device)
        branch = torch.rand(n, device=self.device)
        cfg = self.settings.commands.sampling
        stop_ids = env_ids[branch < cfg.stop]
        self.commands[stop_ids] = 0.
        self.command_time_left[stop_ids] = torch_rand_float(3., 5., (len(stop_ids),), self.device)
        lo = cfg.stop
        for axis, probability, key in ((0, cfg.x, 'lin_vel_x'), (1, cfg.y, 'lin_vel_y'), (2, cfg.yaw, 'ang_vel_yaw')):
            ids = env_ids[(branch >= lo) & (branch < lo + probability)]
            lo += probability
            self.commands[ids] = 0.
            sign = torch.where(torch.rand(len(ids), device=self.device) < .5, -1., 1.)
            magnitude = torch.where(sign > 0, ranges[key][1], -ranges[key][0])
            self.commands[ids, axis] = sign * magnitude * torch_rand_float(.6, 1., (len(ids),), self.device)
            if axis == 2:
                self.command_time_left[ids] = torch_rand_float(4., 6., (len(ids),), self.device)

    def _compute_torques(self, actions):
        if not self.is_x5:
            return super()._compute_torques(actions)
        cfg = self.settings.control
        target = (self.default_dof_pos[:, :12] + cfg.action_scale * actions[:, :12]
                  + self.motor_offset[:, :12]).clamp(self.leg_target_min, self.leg_target_max)
        requested = torch.zeros_like(actions)
        requested[:, :12] = (self.p_gains[:12] * self.Kp_factors * (target-self.dof_pos[:, :12])
                            - self.d_gains[:12] * self.Kd_factors * self.dof_vel[:, :12])
        self.wheel_policy_target = (actions[:, 12:] * cfg.vel_scale).clamp(-cfg.wheel_target_limit, cfg.wheel_target_limit)
        wheel_target = self.wheel_policy_target
        pi = cfg.stop_pi
        physics_dt = self.settings.sim.dt
        forces = self.contact_sensor.data.net_forces_w[:, self.sensor_ids][:, self.feet_indices]
        contact = forces.norm(dim=-1) > pi.contact_force_n
        self.stop_contact_time = torch.where(contact, self.stop_contact_time + physics_dt, 0.)
        if pi.enabled:
            xy, yaw = self.commands[:, :2].norm(dim=-1), self.commands[:, 2].abs()
            enter = (xy < pi.enter_linear) & (yaw < pi.enter_yaw)
            leave = (xy >= pi.exit_linear) | (yaw >= pi.exit_yaw)
            self.stop_requested = (self.stop_requested | enter) & ~leave
            self.stop_control_time = torch.where(self.stop_requested, self.stop_control_time + physics_dt, 0.)
            blend = smooth_gate(self.stop_control_time, (pi.dwell_s, pi.dwell_s + pi.ramp_s))
            wheel_target = wheel_target * (1-blend[:, None])
        error = wheel_target - self.dof_vel[:, 12:]
        proportional = self.d_gains[12:] * self.Kd_factors * error
        if pi.enabled:
            eligible = ((self.stop_control_time >= pi.dwell_s + pi.ramp_s)[:, None]
                        & contact & (self.stop_contact_time >= pi.contact_dwell_s))
            self.stop_integral *= eligible
            total = proportional + self.stop_integral
            integrate = eligible & ((total.abs() < pi.torque_limit) | (error * total <= 0))
            self.stop_integral = (self.stop_integral + integrate * pi.ki * error * physics_dt).clamp(
                -pi.integral_limit, pi.integral_limit)
            proportional = (proportional + self.stop_integral).clamp(-pi.torque_limit, pi.torque_limit)
        requested[:, 12:] = proportional
        self.motor_violation = self.motor_limiter.violation_count(requested, self.dof_pos, self.dof_vel).float()
        return self.motor_limiter.clip(requested, self.dof_pos, self.dof_vel)

    def compute_reward(self):
        if self.is_x5:
            self._update_x5_scans()
            # Update once, even in S2, so logging/reset never consumes stale state.
            self.x5_motion.update()
        return super().compute_reward()

    def _reward_weight(self, name):
        if not self.is_x5:
            return super()._reward_weight(name)
        if name == 'lin_vel_z':
            cfg = self.settings.rewards.lin_vel_z_curriculum
            fraction = min(1., max(0., (self.training_iteration-cfg.start_iteration)
                                  / (cfg.end_iteration-cfg.start_iteration)))
            start = self.settings.rewards.scales[name]
            return (start + fraction*(cfg.final_weight-start))*self.dt
        return self.reward_scales[name]

    def _reward_joint_pose_corridor(self):
        return self.x5_motion.costs['joint_pose']

    def _reward_wheel_position_corridor(self):
        return self.x5_motion.costs['wheel_position']

    def _reward_wheelbase_corridor(self):
        return self.x5_motion.costs['wheelbase']

    def _reward_track_width_corridor(self):
        return self.x5_motion.costs['track_width']

    def _reward_height_corridor(self):
        return self.x5_motion.costs['height']

    def _reward_maneuver_completed_step(self):
        return self.x5_motion.costs['maneuver_step']

    def _reward_terrain_roll_pitch_l2(self):
        return self.x5_terrain.tilt_cost

    def _reward_lateral_progress(self):
        desired = self.commands[:, 1]
        progress = (self.base_lin_vel[:, 1]*desired.sign()/desired.abs().clamp_min(.08)).clamp(0, 1)
        tracking = torch.exp(-((self.base_lin_vel[:, :2]-self.commands[:, :2])/.25).square().sum(-1))
        yaw_tracking = torch.exp(-((self.base_ang_vel[:, 2]-self.commands[:, 2])/.5).square())
        return (desired.abs() > .08)*progress*tracking*yaw_tracking

    def _stop_reward_gate(self):
        cfg = self.settings.control.stop_pi
        return (self.commands[:, :2].norm(dim=-1) < cfg.enter_linear) & (self.commands[:, 2].abs() < cfg.enter_yaw)

    def _reward_stop_wheel_speed_l2(self):
        return self._stop_reward_gate() * (self.dof_vel[:, 12:] * self.settings.mujica.wheel_radius/.2).square().mean(-1)

    def _reward_stop_wheel_target_l2(self):
        return self._stop_reward_gate() * (self.wheel_policy_target * self.settings.mujica.wheel_radius/.5).square().mean(-1)

    def _reward_joint_power(self):
        return (self.dof_vel * self.applied_torques).abs().sum(-1)

    def _reward_action_smoothness(self):
        difference = self.actions-2*self.last_actions+self.last_last_actions
        return (difference.square()*(self.last_actions != 0)*(self.last_last_actions != 0)).sum(-1)

    def _reward_undesired_contacts(self):
        return (self.contact_force_history[:, :, self.penalised_contact_indices].norm(dim=-1).amax(dim=1) > 5.).sum(-1)

    def _reward_joint_pos_limits(self):
        q = self.dof_pos[:, :12]
        return ((self.soft_joint_limits[:, 0]-q).clamp_min(0) + (q-self.soft_joint_limits[:, 1]).clamp_min(0)).sum(-1)

    def _reward_leg_joint_acc_l2(self):
        return self.dof_acc[:, :12].square().sum(-1)

    def _reward_wheel_joint_acc_l2(self):
        return self.dof_acc[:, 12:].square().sum(-1)

    def _reward_leg_joint_torques_l2(self):
        return self.applied_torques[:, :12].square().sum(-1)

    def _reward_wheel_joint_torques_l2(self):
        return self.applied_torques[:, 12:].square().sum(-1)

    def _reset_x5(self, env_ids):
        metrics = self.x5_motion.reset(env_ids)
        self.x5_terrain.reset(env_ids)
        for buffer in (self.stop_integral, self.stop_contact_time, self.stop_control_time,
                       self.stop_requested, self.wheel_policy_target, self.motor_offset):
            buffer[env_ids] = 0
        if self.settings.domain_rand.randomize_motor_offset:
            self.motor_offset[env_ids, :12] = torch_rand_float(*self.settings.domain_rand.motor_offset_range,
                                                               (len(env_ids), 12), self.device)
        return metrics
