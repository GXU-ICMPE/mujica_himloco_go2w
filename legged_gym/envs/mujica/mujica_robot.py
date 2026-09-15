"""Multi-task S1 and selector-controlled S2 for the HIMLoco Go2-W asset.

Skill IDs: 0 flat/slopes, 1 discrete obstacles, 2 stairs. Assignment is curriculum metadata
only in S1. In S2, changing a selected skill never changes the physical task,
commands, episode clock, reset distribution or termination rules.
"""
import torch
from mujica.locomotion_tasks import LocomotionTasks
from mujica.skills import skill_metadata
from mujica.terrain import assign_terrain_columns
from mujica.rewards import reward_metadata, validate_reward_settings
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_from_euler_xyz, quat_rotate_inverse, torch_rand_float
from legged_gym.envs.go2w.go2w_robot import Go2w
from mujica.motor import DCMotorLimiter
from .mujica_terrain import MUJICATerrain


# The supplied URDF retains both head fixed joints (dont_collapse=true).
# 19 physical bodies -> 18 real contact groups by grouping the two head bodies.
# No fake padded contact labels are used. Other fixed collision shapes are
# merged into their calf/base body by the original asset loader.
COLLISION_GROUPS = (("base",), ("Head_upper", "Head_lower")) + tuple(
    (f"{leg}_{segment}",) for leg in ("FL", "FR", "RL", "RR")
    for segment in ("hip", "thigh", "calf", "foot"))


class MUJICARobot(LocomotionTasks, Go2w):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.settings = cfg
        validate_reward_settings(cfg.rewards, cfg.terrain)
        self.stage = cfg.mujica.stage
        if self.stage not in ("s1", "s2"):
            raise ValueError("cfg.mujica.stage must be s1 or s2")
        if cfg.env.num_one_step_observations != 58 or cfg.env.num_observations != 348:
            raise ValueError("MUJICA requires 6 x 58 actor observations")
        if cfg.env.num_privileged_obs != 270 or not cfg.terrain.measure_heights:
            raise ValueError("MUJICA requires 270 privileged observations and 187 heights")
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.reset()

    def create_sim(self):
        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id,
                                       self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ("heightfield", "trimesh"):
            self.terrain = MUJICATerrain(self.cfg.terrain, self.num_envs)
            if mesh_type == "heightfield":
                self._create_heightfield()
            else:
                self._create_trimesh()
        elif mesh_type == "plane":
            self._create_ground_plane()
        else:
            raise ValueError("MUJICA supports plane, heightfield or trimesh")
        self._create_envs()

    def _get_env_origins(self):
        super()._get_env_origins()
        if self.custom_origins:
            self.terrain_types = torch.tensor(assign_terrain_columns(self.num_envs, self.terrain.task_ids), device=self.device)
            self.env_origins = self.terrain_origins[self.terrain_levels, self.terrain_types].clone()
            assignments = torch.as_tensor(self.terrain.task_ids, device=self.device)
            self.task_ids = assignments[self.terrain_types].long()
        else:
            self.task_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _create_envs(self):
        super()._create_envs()
        body_names = list(self.gym.get_actor_rigid_body_names(self.envs[0], self.actor_handles[0]))
        self.body_names = body_names
        missing = [name for group in COLLISION_GROUPS for name in group if name not in body_names]
        if missing:
            raise RuntimeError("Missing Go2-W collision bodies: " + str(missing) +
                               "; keep Head_upper/Head_lower fixed joints uncollapsed")
        self.collision_body_groups = [torch.tensor([body_names.index(name) for name in group],
                                                   dtype=torch.long, device=self.device)
                                      for group in COLLISION_GROUPS]
        if len(self.feet_indices) != 4 or len(self.wheel_indices) != 4 or self.num_dof != 16:
            raise RuntimeError("MUJICA Go2-W requires exactly four wheels and 16 actuated DOFs")
        # Keep stable FL,FR,RL,RR order for the four supervised clearance targets.
        self.clearance_body_indices = torch.tensor([body_names.index(f"{leg}_foot")
            for leg in ("FL", "FR", "RL", "RR")], device=self.device)

    def _init_buffers(self):
        # Upstream initializes new friction buffers after randomizing real actors;
        # retain the actual values so later resets do not report stale parameters.
        physical_friction = getattr(self, "friction_coeffs", None)
        physical_restitution = getattr(self, "restitution_coeffs", None)
        super()._init_buffers()
        if physical_friction is not None:
            self.friction_coeffs = physical_friction
        if physical_restitution is not None:
            self.restitution_coeffs = physical_restitution
        self.skill_ids = self.task_ids.clone() if self.stage == "s1" else torch.zeros_like(self.task_ids)
        self._pending_reset = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_start_xy = self.root_states[:, :2].clone()
        self.command_distance = torch.zeros(self.num_envs, device=self.device)
        self.tracking_streak = torch.zeros(self.num_envs, device=self.device)
        self.tracking_best_streak = torch.zeros(self.num_envs, device=self.device)
        self.failure_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.terrain_exit_buf = torch.zeros_like(self.failure_buf)
        self.motor_violation = torch.zeros(self.num_envs, device=self.device)
        self.motor_limiter = DCMotorLimiter(self.dof_names, self.torque_limits,
                                           self.dof_vel_limits, self.cfg.mujica.motor)
        if self.num_height_points != 187:
            raise ValueError("MUJICA expects the upstream 17 x 11 height grid")

    def _get_noise_scale_vec(self, cfg):
        # Skill, command and action channels must not receive observation noise.
        self.add_noise = cfg.noise.add_noise
        v = torch.zeros(58, device=self.device)
        s, n, k = cfg.noise.noise_scales, cfg.noise.noise_level, self.obs_scales
        v[0:3] = s.ang_vel * n * k.ang_vel
        v[3:6] = s.gravity * n
        v[9:25] = s.dof_pos * n * k.dof_pos
        v[25:41] = s.dof_vel * n * k.dof_vel
        return v

    @property
    def selector_obs(self):
        return self.obs_buf.view(self.num_envs, 6, 58)[:, :, :57].reshape(self.num_envs, 342)

    def get_selector_observations(self):
        return self.selector_obs

    def set_skill(self, ids):
        """Set current scalar skill; never append a frame or reset a robot.

        Updating historical skill labels would rewrite the experienced history,
        so only the newest observation and corresponding privileged frame change.
        """
        ids = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        if ids.ndim == 0:
            ids = ids.expand(self.num_envs)
        if ids.shape != (self.num_envs,) or bool(((ids < 0) | (ids > 2)).any()):
            raise ValueError("skill IDs must have shape [num_envs] with values in {0,1,2}")
        self.skill_ids.copy_(ids)
        self.obs_buf[:, 57] = ids.float()
        self.privileged_obs_buf[:, 57] = ids.float()

    def _refresh_kinematics(self):
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        states = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)
        self.feet_pos = states[:, self.feet_indices, :3]
        self.feet_vel = states[:, self.feet_indices, 7:10]

    def _sample_ground_at(self, xyz):
        if self.cfg.terrain.mesh_type == "plane":
            return torch.zeros_like(xyz[..., 2])
        pixels = ((xyz[..., :2] + self.cfg.terrain.border_size) /
                  self.cfg.terrain.horizontal_scale).long()
        x = pixels[..., 0].clamp(0, self.height_samples.shape[0]-2)
        y = pixels[..., 1].clamp(0, self.height_samples.shape[1]-2)
        h = torch.minimum(torch.minimum(self.height_samples[x, y],
                                       self.height_samples[x+1, y]), self.height_samples[x, y+1])
        return h * self.cfg.terrain.vertical_scale

    def _current_frame(self):
        # Wheel coordinates are omitted in an observation COPY. Never write to
        # dof_pos/dof_vel: these are views into the live Isaac Gym state tensor.
        q = self.dof_pos - self.default_dof_pos
        q[:, self.wheel_indices] = 0.0
        self.dof_err = q
        obs = torch.cat((self.base_ang_vel*self.obs_scales.ang_vel,
                         self.projected_gravity,
                         self.commands[:, :3]*self.commands_scale,
                         q*self.obs_scales.dof_pos,
                         self.dof_vel*self.obs_scales.dof_vel,
                         self.actions,
                         self.skill_ids.float().unsqueeze(1)), dim=-1)
        if self.add_noise:
            obs = obs + (2*torch.rand_like(obs)-1)*self.noise_scale_vec
        magnitude = torch.linalg.vector_norm(self.contact_forces, dim=-1)
        collision = torch.stack([(magnitude[:, group] > self.cfg.mujica.contact_threshold).any(dim=1)
                                 for group in self.collision_body_groups], dim=1).float()
        states = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)
        wheel_xyz = states[:, self.clearance_body_indices, :3]
        # Vertical clearance from the lower wheel surface to the heightfield.
        # This radius correction is a documented approximation on tilted wheels.
        clearance = (wheel_xyz[..., 2] - self._sample_ground_at(wheel_xyz) -
                     self.cfg.mujica.wheel_radius).clamp(min=0.0)
        heights = (self.root_states[:, 2:3] - 0.5 - self.measured_heights).clamp(-1, 1)
        heights = heights * self.obs_scales.height_measurements
        privileged = torch.cat((obs, self.base_lin_vel*self.obs_scales.lin_vel,
                                collision, clearance, heights), dim=-1)
        clip = self.cfg.normalization.clip_observations
        return obs.clamp(-clip, clip), privileged.clamp(-clip, clip)

    def compute_observations(self):
        frame, privileged = self._current_frame()
        self.obs_buf = torch.cat((frame, self.obs_buf[:, :-58]), dim=-1)
        self.privileged_obs_buf = privileged
        if self._pending_reset.any():
            ids = self._pending_reset.nonzero(as_tuple=False).flatten()
            self.obs_buf[ids] = frame[ids].repeat(1, 6)
            self._pending_reset[ids] = False

    def get_current_obs(self):
        return self._current_frame()[1]

    def compute_termination_observations(self, env_ids):
        return self._current_frame()[1][env_ids]

    def reset(self):
        # Do not consume a hidden policy step during reset.
        ids = torch.arange(self.num_envs, device=self.device)
        self.reset_idx(ids)
        self.compute_observations()
        return self.obs_buf, self.privileged_obs_buf

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        # Capture outcomes before the parent clears episode sums and updates
        # terrain level/origin. This is diagnostics only, not reward shaping.
        skill_metrics = self._episode_skill_metrics(env_ids)
        super().reset_idx(env_ids)
        self.extras["episode"].update(skill_metrics)
        self._reset_reward_diagnostics(env_ids)
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.last_last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.last_root_vel[env_ids] = 0.0
        self.last_contacts[env_ids] = False
        self.contact_forces[env_ids] = 0.0
        self._pending_reset[env_ids] = True
        self.episode_start_xy[env_ids] = self.root_states[env_ids, :2]
        self.command_distance[env_ids] = 0.0
        self.tracking_streak[env_ids] = 0.0
        self.tracking_best_streak[env_ids] = 0.0
        if self.stage == "s1":
            self.skill_ids[env_ids] = self.task_ids[env_ids]
        self._refresh_kinematics()
        # Gym rigid-body tensors are only propagated by physics. Reconstruct
        # post-reset wheel positions from URDF FK instead of using terminal data.
        self._reset_wheel_kinematics(env_ids)
        self.measured_heights = self._get_heights()


    def _reset_wheel_kinematics(self, env_ids):
        from isaacgym.torch_utils import quat_apply
        states = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)
        # Analytic kinematics for the bundled Go2-W URDF. Fixed hip offsets and
        # link lengths are read once from the asset rather than inferred indices.
        if not hasattr(self, "_fk_description"):
            import os
            import xml.etree.ElementTree as ET
            from legged_gym import LEGGED_GYM_ROOT_DIR
            path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            root = ET.parse(os.path.abspath(path)).getroot()
            self._fk_description = {}
            for joint in root.findall("joint"):
                origin = joint.find("origin")
                xyz = [float(x) for x in origin.get("xyz", "0 0 0").split()]
                rpy = [float(x) for x in origin.get("rpy", "0 0 0").split()]
                axis = joint.find("axis")
                self._fk_description[joint.get("name")] = (xyz, rpy,
                    [float(x) for x in axis.get("xyz", "1 0 0").split()] if axis is not None else [1., 0., 0.])
        from isaacgym.torch_utils import quat_mul, quat_from_angle_axis
        n = len(env_ids)
        for leg, body_idx in zip(("FL", "FR", "RL", "RR"), self.clearance_body_indices):
            pos = self.root_states[env_ids, :3].clone()
            quat = self.root_states[env_ids, 3:7].clone()
            for segment in ("hip", "thigh", "calf", "foot"):
                name = f"{leg}_{segment}_joint"
                xyz, rpy, axis = self._fk_description[name]
                offset = torch.tensor(xyz, device=self.device).expand(n, -1)
                pos += quat_apply(quat, offset)
                roll, pitch, yaw = (torch.full((n,), v, device=self.device) for v in rpy)
                quat = quat_mul(quat, quat_from_euler_xyz(roll, pitch, yaw))
                q = self.dof_pos[env_ids, self.dof_names.index(name)]
                axis_t = torch.tensor(axis, device=self.device).expand(n, -1)
                quat = quat_mul(quat, quat_from_angle_axis(q, axis_t))
            states[env_ids, body_idx, :3] = pos
            states[env_ids, body_idx, 3:7] = quat
            states[env_ids, body_idx, 7:13] = 0.0
        self.feet_pos = states[:, self.feet_indices, :3]
        self.feet_vel = states[:, self.feet_indices, 7:10]

    def _reset_root_states(self, env_ids):
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # All skills start upright within the flat central spawn patch.
        self.root_states[env_ids, :2] += torch_rand_float(-0.25, 0.25, (len(env_ids), 2), self.device)
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), self.device)
        ids_i32 = env_ids.to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(ids_i32), len(ids_i32))


    def _reward_dof_vel(self):
        # Same mathematical Go2-W reward, without corrupting live wheel states.
        vel = self.dof_vel.clone()
        vel[:, self.wheel_indices] = 0.0
        return torch.sum(torch.square(vel), dim=1)

    def post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self._refresh_kinematics()
        self._post_physics_step_callback()
        self._refresh_kinematics()
        self._update_progress_metrics()
        self.check_termination()
        self.compute_reward()
        # One observation sample: terminal successors and regular successors
        # refer to exactly the same physical state and observation-noise draw.
        self.compute_observations()
        next_obs = self.obs_buf.clone()
        next_privileged = self.privileged_obs_buf.clone()
        reset_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        terminal_privileged = next_privileged[reset_ids].clone()
        self.extras = {"time_outs": self.time_out_buf.clone(),
                       "next_observations": next_obs,
                       "next_privileged_observations": next_privileged,
                       "motor_violation_count": self.motor_violation.clone()}
        self.last_last_actions[:] = self.last_actions
        self.last_actions[:] = self.actions
        self.last_dof_vel[:] = self.dof_vel
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.disturbance.zero_()
        self.reset_idx(reset_ids)
        if len(reset_ids):
            # Replace only reset rows. Appending all rows would advance each
            # non-terminal history twice within one control tick.
            reset_frame, reset_privileged = self._current_frame()
            self.obs_buf[reset_ids] = reset_frame[reset_ids].repeat(1, 6)
            self.privileged_obs_buf[reset_ids] = reset_privileged[reset_ids]
            self._pending_reset[reset_ids] = False
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()
        return reset_ids, terminal_privileged

    def export_metadata(self):
        def values(tensor):
            return tensor.detach().cpu().flatten().tolist()
        return {
            **skill_metadata(),
            "reward_profile": reward_metadata(self.settings.rewards),
            "simulator": "isaacgym",
            "joint_names": list(self.dof_names),
            "wheel_indices": values(self.wheel_indices),
            "default_dof_pos": values(self.default_dof_pos),
            "p_gains": values(self.p_gains), "d_gains": values(self.d_gains),
            "torque_limits": values(self.torque_limits),
            "velocity_limits": values(self.dof_vel_limits),
            "action_scale": self.cfg.control.action_scale,
            "vel_scale": self.cfg.control.vel_scale,
            "clip_actions": self.cfg.normalization.clip_actions,
            "control_dt": self.dt, "sim_dt": self.sim_params.dt,
            "obs_scales": {name: getattr(self.obs_scales, name)
                for name in ("ang_vel", "lin_vel", "dof_pos", "dof_vel")},
            "motor": dict(self.motor_limiter.config),
            "collision_groups": [list(group) for group in COLLISION_GROUPS],
            "clearance_order": ["FL", "FR", "RL", "RR"],
            "wheel_radius": self.cfg.mujica.wheel_radius,
            "history_order": "newest_first",
        }
