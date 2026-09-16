"""MUJICA S1/S2 on native Isaac Lab scene, articulation and contact APIs."""
import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.terrains.utils import create_prim_from_mesh

from mujica.motor import DCMotorLimiter
from .math import torch_rand_float, yaw_rotate
from .settings import joint_contract, joint_indices, settings_from_dict
from .robots import robot_spec
from .x5_task import X5Task
from .task_logic import TaskLogic
from .terrain import HeightField
from mujica.terrain import assign_terrain_columns
from mujica.skills import skill_metadata
from mujica.rewards import reward_metadata


class MUJICAEnv(X5Task, TaskLogic, DirectRLEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        self.settings = settings_from_dict(cfg.task_settings)
        self.stage = self.settings.mujica.stage
        self.spec = robot_spec(self.settings.asset.name)
        self.is_x5 = self.spec.name == 'x5'
        self.urdf_path = self.spec.urdf
        super().__init__(cfg, render_mode, **kwargs)
        self.dt = self.step_dt
        self.num_actions = self.num_dof = 16
        self.dof_names = list(self.spec.joints)
        self.joint_ids = torch.tensor(joint_indices(self.robot.joint_names, self.spec.name), device=self.device)
        self.body_names = list(self.robot.body_names)
        self.num_bodies = len(self.body_names)
        expected = {name for group in self.spec.collision_groups for name in group}
        if set(self.body_names) != expected:
            raise RuntimeError(f"Expected named {self.spec.name} bodies {sorted(expected)}; imported {self.body_names}")
        self.collision_body_groups = [torch.tensor([self.body_names.index(n) for n in group], device=self.device)
                                      for group in self.spec.collision_groups]
        self.sensor_ids = torch.tensor([self.contact_sensor.body_names.index(n) for n in self.body_names], device=self.device)
        self.clearance_body_indices = torch.tensor([self.body_names.index(n) for n in self.spec.feet], device=self.device)
        self.feet_indices = self.clearance_body_indices
        self.wheel_indices = torch.tensor([self.dof_names.index(n) for n in self.spec.wheels], device=self.device)
        self.hip_indices = torch.tensor([self.dof_names.index(n) for n in self.spec.hips], device=self.device)
        self.leg_indices = torch.tensor([i for i, n in enumerate(self.dof_names) if n not in self.spec.wheels], device=self.device)
        self.base_index = self.body_names.index("base")
        self.penalised_contact_indices = torch.tensor([i for i, n in enumerate(self.body_names)
            if any(part in n for part in self.settings.asset.penalize_contacts_on)], device=self.device)
        self.termination_contact_indices = torch.tensor([i for i, n in enumerate(self.body_names)
            if any(part in n for part in self.settings.asset.terminate_after_contacts_on)], device=self.device)
        self._initialize_buffers()
        if self.is_x5:
            self._initialize_x5()
        self._randomize_body_properties()
        self.init_done = True

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contacts"] = self.contact_sensor
        t = self.settings.terrain
        material = sim_utils.RigidBodyMaterialCfg(static_friction=t.static_friction,
            dynamic_friction=t.dynamic_friction, restitution=t.restitution,
            friction_combine_mode="average", restitution_combine_mode="average")
        self.custom_origins = t.mesh_type == "trimesh"
        if self.custom_origins:
            self.terrain = HeightField(t)
            create_prim_from_mesh("/World/ground", self.terrain.mesh(), physics_material=material,
                                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)))
        else:
            sim_utils.spawn_ground_plane("/World/ground", sim_utils.GroundPlaneCfg(physics_material=material))
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

    def _initialize_buffers(self):
        s = self.settings
        def zeros(*shape, dtype=torch.float):
            return torch.zeros(self.num_envs, *shape, device=self.device, dtype=dtype)
        self.default_dof_pos = torch.tensor([[s.init_state.default_joint_angles[n] for n in self.dof_names]], device=self.device)
        self.p_gains = torch.tensor([next(v for k, v in s.control.stiffness.items() if k in n) for n in self.dof_names], device=self.device)
        self.d_gains = torch.tensor([next(v for k, v in s.control.damping.items() if k in n) for n in self.dof_names], device=self.device)
        limits = joint_contract(robot=self.spec.name)
        self.torque_limits = torch.tensor([limits[n]["effort"] for n in self.dof_names], device=self.device)
        self.dof_vel_limits = torch.tensor([limits[n]["velocity"] for n in self.dof_names], device=self.device)
        pos_limits = torch.tensor([[limits[self.dof_names[i]][key] for key in ('lower', 'upper')]
                                   for i in self.leg_indices.tolist()], device=self.device)
        center = pos_limits.mean(-1, keepdim=True)
        self.soft_joint_limits = center + (pos_limits-center)*s.rewards.soft_dof_pos_limit
        self.motor_limiter = DCMotorLimiter(self.dof_names, self.torque_limits, self.dof_vel_limits, s.mujica.motor)
        self.Kp_factors, self.Kd_factors, self.motor_strength_factors = (zeros(1)+1 for _ in range(3))
        self.actions, self.last_actions, self.last_last_actions = (zeros(16) for _ in range(3))
        self.torques, self.last_dof_vel, self.last_root_vel = zeros(16), zeros(16), zeros(6)
        self.commands = zeros(4)
        self.command_ranges = s.commands.ranges
        self.obs_scales = s.normalization.obs_scales
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel]*2 + [self.obs_scales.ang_vel], device=self.device)
        self.noise_scale_vec = self._get_noise_scale_vec(s)
        self.gravity_vec = zeros(3)
        self.gravity_vec[:, 2] = -1
        self.base_lin_vel, self.base_ang_vel, self.projected_gravity = (zeros(3) for _ in range(3))
        self.history_buf, self.privileged_obs_buf = zeros(348), zeros(self.spec.critic_dim)
        self._pending_reset = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._successor_ready = False
        self.rew_buf, self.motor_violation = zeros(), zeros()
        self.command_distance, self.tracking_streak, self.tracking_best_streak = (zeros() for _ in range(3))
        self.failure_buf, self.terrain_exit_buf = (zeros(dtype=torch.bool) for _ in range(2))
        self.episode_start_xy = zeros(2)
        self.last_contacts = zeros(4, dtype=torch.bool)
        self.reward_scales = {k: v*self.dt for k, v in s.rewards.scales.items() if v != 0}
        if any(not hasattr(self, "_reward_"+name) for name in self.reward_scales):
            raise ValueError("Unsupported reward term in saved environment")
        self.episode_sums = {k: zeros() for k in self.reward_scales}
        self.height_points = torch.stack(torch.meshgrid(
            torch.tensor(s.terrain.measured_points_x, device=self.device),
            torch.tensor(s.terrain.measured_points_y, device=self.device), indexing="ij"), -1).reshape(1, 187, 2)
        self.height_points = torch.cat((self.height_points, torch.zeros(1, 187, 1, device=self.device)), -1)
        self.num_height_points = 187
        if self.custom_origins:
            self.height_samples = torch.tensor(self.terrain.height_field_raw, device=self.device)
            self.terrain_origins = torch.tensor(self.terrain.env_origins, device=self.device)
            self.terrain_types = torch.tensor(assign_terrain_columns(self.num_envs, self.terrain.task_ids), device=self.device)
            self.max_terrain_level = s.terrain.num_rows
            self.terrain_levels = torch.randint(min(s.terrain.max_init_terrain_level+1, s.terrain.num_rows), (self.num_envs,), device=self.device)
            self.env_origins = self.terrain_origins[self.terrain_levels, self.terrain_types].clone()
            assignments = torch.tensor(self.terrain.task_ids, device=self.device)
            self.task_ids = assignments[self.terrain_types]
        else:
            self.env_origins = self.scene.env_origins.clone()
            self.task_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.skill_ids = self.task_ids.clone() if self.stage == "s1" else torch.zeros_like(self.task_ids)
        self._refresh_state()
        self.measured_heights = self._get_heights()

    def _refresh_state(self):
        # Explicit copies keep observation zeroing/reset bookkeeping out of Lab caches.
        self.root_states = self.robot.data.root_state_w.clone()
        self.base_quat = self.root_states[:, 3:7]
        self.dof_pos = self.robot.data.joint_pos[:, self.joint_ids].clone()
        self.dof_vel = self.robot.data.joint_vel[:, self.joint_ids].clone()
        self.rigid_body_states = (self.robot.data.body_link_state_w if self.is_x5 else self.robot.data.body_state_w).clone()
        self.contact_forces = self.contact_sensor.data.net_forces_w[:, self.sensor_ids].clone()
        if self.is_x5:
            self.dof_acc = self.robot.data.joint_acc[:, self.joint_ids].clone()
            self.applied_torques = self.robot.data.applied_torque[:, self.joint_ids].clone()
            self.contact_force_history = self.contact_sensor.data.net_forces_w_history[:, :, self.sensor_ids].clone()
        self._refresh_kinematics()

    def _get_heights(self):
        points = yaw_rotate(self.base_quat, self.height_points) + self.root_states[:, None, :3]
        return self._sample_ground_at(points)

    def _randomize_body_properties(self):
        s = self.settings.domain_rand
        view = self.robot.root_physx_view
        ids = torch.arange(self.num_envs, dtype=torch.int, device="cpu")
        masses = view.get_masses().clone()
        before = masses.clone()
        if s.randomize_payload_mass:
            masses[:, self.base_index] += torch_rand_float(*s.payload_mass_range, (self.num_envs,), "cpu")
        if s.randomize_link_mass:
            other = [i for i in range(self.num_bodies) if i != self.base_index]
            masses[:, other] *= torch_rand_float(*s.link_mass_range, (self.num_envs, len(other)), "cpu")
        if torch.any(masses <= 0):
            raise ValueError("Randomized rigid body mass must be positive")
        if s.randomize_payload_mass or s.randomize_link_mass:
            view.set_masses(masses, ids)
            # Match Gym's recomputeInertia behavior when changing body mass.
            inertia = view.get_inertias().clone() * (masses/before).unsqueeze(-1)
            view.set_inertias(inertia, ids)
        if s.randomize_com_displacement:
            coms = view.get_coms().clone()
            # The bundled Gym code assigns this range as the base COM, rather than adding it.
            displacement = torch_rand_float(*s.com_displacement_range, (self.num_envs, 3), "cpu")
            if self.is_x5:
                coms[:, self.base_index, :3] += displacement
            else:
                coms[:, self.base_index, :3] = displacement
            view.set_coms(coms, ids)

    def _randomize_reset_properties(self, env_ids):
        s = self.settings.domain_rand
        for enabled, limits, buffer in ((s.randomize_kp, s.kp_range, self.Kp_factors),
                (s.randomize_kd, s.kd_range, self.Kd_factors),
                (s.randomize_motor_strength, s.motor_strength_range, self.motor_strength_factors)):
            if enabled:
                buffer[env_ids] = torch_rand_float(*limits, (len(env_ids), 1), self.device)
        # One material per robot, applied to every collision shape as in Gym.
        ids = env_ids.to(device="cpu", dtype=torch.int)
        materials = self.robot.root_physx_view.get_material_properties().clone()
        friction = (torch_rand_float(*s.friction_range, (len(ids), 1), "cpu") if s.randomize_friction
                    else torch.full((len(ids), 1), self.settings.terrain.static_friction))
        restitution = (torch_rand_float(*s.restitution_range, (len(ids), 1), "cpu") if s.randomize_restitution
                       else torch.full((len(ids), 1), self.settings.terrain.restitution))
        materials[ids, :, 0] = friction
        materials[ids, :, 1] = friction
        materials[ids, :, 2] = restitution
        self.robot.root_physx_view.set_material_properties(materials, ids)

    def _pre_physics_step(self, actions):
        self.extras = {}
        self._successor_ready = False
        clip = self.settings.normalization.clip_actions
        self.actions = actions.clamp(-clip, clip).clone()
        self._action_substep = 0
        self._delay_steps = (torch.randint(self.cfg.decimation, (self.num_envs, 1), device=self.device)
                             if self.settings.domain_rand.delay else torch.zeros(self.num_envs, 1, device=self.device))
        # Gym's force API applies a one-physics-step pulse after every 8 control
        # steps (the legacy field is a step count, not seconds).
        self._disturbance = torch.zeros(self.num_envs, 1, 3, device=self.device)
        d = self.settings.domain_rand
        if d.disturbance and self.common_step_counter > 0 and self.common_step_counter % d.disturbance_interval == 0:
            self._disturbance = torch_rand_float(*d.disturbance_range, (self.num_envs, 1, 3), self.device)

    def _apply_action(self):
        self.dof_pos = self.robot.data.joint_pos[:, self.joint_ids]
        self.dof_vel = self.robot.data.joint_vel[:, self.joint_ids]
        delayed = torch.where(self._action_substep >= self._delay_steps, self.actions, self.last_actions)
        self.torques = self._compute_torques(delayed)
        self.robot.set_joint_effort_target(self.torques, joint_ids=self.joint_ids)
        force = self._disturbance if self._action_substep == 0 else torch.zeros_like(self._disturbance)
        self.robot.permanent_wrench_composer.set_forces_and_torques(
            forces=force, torques=torch.zeros_like(force), body_ids=[self.base_index], is_global=False)
        self._action_substep += 1

    def _get_dones(self):
        self._refresh_state()
        if not self.is_x5:
            interval = max(1, int(self.settings.commands.resampling_time/self.dt))
            self._resample_commands((self.episode_length_buf % interval == 0).nonzero().flatten())
        d = self.settings.domain_rand
        if d.push_robots and self.common_step_counter % max(1, math.ceil(d.push_interval_s/self.dt)) == 0:
            velocity = self.root_states[:, 7:13].clone()
            delta = torch_rand_float(-d.max_push_vel_xy, d.max_push_vel_xy, (self.num_envs, 2), self.device)
            if self.is_x5:
                velocity[:, :2] += delta
                velocity[:, 5] += torch_rand_float(-.15, .15, (self.num_envs,), self.device)
            else:
                velocity[:, :2] = delta
            self.robot.write_root_velocity_to_sim(velocity)
            self.root_states[:, 7:13] = velocity
            self._refresh_kinematics()
        self.measured_heights = self._get_heights()
        self._update_progress_metrics()
        self.check_termination()
        return self.reset_buf & ~self.time_out_buf, self.time_out_buf

    def _get_rewards(self):
        self.compute_reward()
        if self.is_x5:
            # Score the command that produced this action; present the next
            # command in the successor observation (never change it mid-action).
            self.command_time_left -= self.dt
            self._resample_commands(((self.command_time_left <= 0) & ~self.reset_buf).nonzero().flatten())
        # DirectRLEnv invokes rewards before auto-reset. Capture this one noise
        # sample and the full true successor here for both GAE and SwAV.
        self.compute_observations()
        ids = self.reset_buf.nonzero().flatten()
        self.extras.update(time_outs=self.time_out_buf.clone(), terminal_ids=ids,
            terminal_critic=self.privileged_obs_buf[ids].clone(),
            next_observations=self.history_buf.clone(),
            next_privileged_observations=self.privileged_obs_buf.clone(),
            motor_violation_count=self.motor_violation.clone())
        self._successor_ready = True
        self.last_last_actions.copy_(self.last_actions)
        self.last_actions.copy_(self.actions)
        self.last_dof_vel.copy_(self.dof_vel)
        self.last_root_vel.copy_(self.root_states[:, 7:13])
        return self.rew_buf.clone()

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return
        metrics = self._episode_skill_metrics(env_ids)
        if self.settings.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        for name, total in self.episode_sums.items():
            metrics["rew_"+name] = (total[env_ids]/self.episode_length_buf[env_ids].clamp(min=1)/self.dt).mean()
            total[env_ids] = 0
        self._reset_reward_diagnostics(env_ids)
        if self.is_x5:
            metrics.update(self._reset_x5(env_ids))
        if self.custom_origins:
            metrics["terrain_level"] = self.terrain_levels.float().mean()
        self.extras["episode"] = metrics
        super()._reset_idx(env_ids)
        self._randomize_reset_properties(env_ids)
        state = self.robot.data.default_root_state[env_ids].clone()
        state[:, :3] += self.env_origins[env_ids]
        state[:, :2] += torch_rand_float(-0.25, 0.25, (len(env_ids), 2), self.device)
        if self.is_x5:
            state[:, 7:13] = torch_rand_float(-.1, .1, (len(env_ids), 6), self.device)
            state[:, 9] = 0.
            q = self.default_dof_pos.expand(len(env_ids), -1).clone()
            if self.settings.domain_rand.randomize_initial_joint_pos:
                q[:, :12] += torch_rand_float(*self.settings.domain_rand.initial_joint_pos_range,
                                              (len(env_ids), 12), self.device)
        else:
            state[:, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), self.device)
            # Preserve the original Gym reset distribution for Go2W.
            q = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), 16), self.device)
        q_sim = self.robot.data.default_joint_pos[env_ids].clone()
        q_sim[:, self.joint_ids] = q
        self.robot.write_root_state_to_sim(state, env_ids)
        self.robot.write_joint_state_to_sim(q_sim, torch.zeros_like(q_sim), env_ids=env_ids)
        self._resample_commands(env_ids)
        if self.stage == "s1":
            self.skill_ids[env_ids] = self.task_ids[env_ids]
        for buffer in (self.actions, self.last_actions, self.last_last_actions, self.last_dof_vel,
                       self.last_root_vel, self.last_contacts, self.command_distance, self.tracking_streak,
                       self.tracking_best_streak):
            buffer[env_ids] = 0
        self._pending_reset[env_ids] = True
        self.root_states[env_ids] = state
        self.dof_pos[env_ids], self.dof_vel[env_ids] = q, 0.0
        self.base_quat = self.root_states[:, 3:7]
        self.contact_forces[env_ids] = 0
        if self.is_x5:
            self.contact_force_history[env_ids] = 0
        self.episode_start_xy[env_ids] = state[:, :2]
        self._refresh_kinematics()
        # Analytic FK avoids reusing stale terminal wheel positions or stepping
        # physics just to initialize observations. It reads the original URDF.
        self._reset_wheel_kinematics(env_ids)
        self.measured_heights = self._get_heights()

    def _get_observations(self):
        if self._pending_reset.any():
            ids = self._pending_reset.nonzero().flatten()
            frame, privileged = self._current_frame()
            self.history_buf[ids] = frame[ids].repeat(1, 6)
            self.privileged_obs_buf[ids] = privileged[ids]
            self._pending_reset[ids] = False
        elif not self._successor_ready:
            self.compute_observations()
        self._successor_ready = True
        return {"policy": self.history_buf, "critic": self.privileged_obs_buf}

    def export_metadata(self):
        metadata = super().export_metadata()
        metadata.update(simulator="isaaclab", quaternion_order="wxyz",
            robot=self.spec.name,
            simulator_joint_names=list(self.robot.joint_names), policy_to_sim_joint_indices=self.joint_ids.cpu().tolist(),
            urdf_path=str(self.urdf_path), prepared_urdf_path=self.cfg.robot.spawn.asset_path,
            disturbance_interval_unit="control_steps")
        metadata.update(skill_metadata())
        metadata["reward_profile"] = reward_metadata(self.settings.rewards)
        if self.custom_origins:
            metadata.update(terrain_column_names=list(self.terrain.terrain_names),
                            terrain_column_tasks=self.terrain.task_ids.tolist())
        return metadata
