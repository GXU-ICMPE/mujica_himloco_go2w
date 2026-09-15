# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

"""Tensor task logic ported from the bundled MUJICA/HIMLoco environment.

All states use wxyz quaternions and the explicit policy joint order. This module
has no simulator imports, allowing contract tests to exercise production logic.
Original HIMLoco reward code retains the repository BSD-3-Clause license.
"""
import torch
from mujica.locomotion_tasks import LocomotionTasks
from .math import quat_apply, quat_rotate_inverse, quat_mul, quat_from_euler_xyz, quat_from_angle_axis, torch_rand_float

COLLISION_GROUPS = (("base",), ("Head_upper", "Head_lower")) + tuple(
    (f"{leg}_{segment}",) for leg in ("FL", "FR", "RL", "RR")
    for segment in ("hip", "thigh", "calf", "foot"))


class TaskLogic(LocomotionTasks):
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
        return self.history_buf.view(self.num_envs, 6, 58)[:, :, :57].reshape(self.num_envs, 342)

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
        self.history_buf[:, 57] = ids.float()
        self.privileged_obs_buf[:, 57] = ids.float()

    def _refresh_kinematics(self):
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        states = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)
        self.feet_pos = states[:, self.feet_indices, :3]
        self.feet_vel = states[:, self.feet_indices, 7:10]

    def _sample_ground_at(self, xyz):
        if self.settings.terrain.mesh_type == "plane":
            return torch.zeros_like(xyz[..., 2])
        pixels = ((xyz[..., :2] + self.settings.terrain.border_size) /
                  self.settings.terrain.horizontal_scale).long()
        x = pixels[..., 0].clamp(0, self.height_samples.shape[0]-2)
        y = pixels[..., 1].clamp(0, self.height_samples.shape[1]-2)
        h = torch.minimum(torch.minimum(self.height_samples[x, y],
                                       self.height_samples[x+1, y]), self.height_samples[x, y+1])
        return h * self.settings.terrain.vertical_scale

    def _current_frame(self):
        # Wheel coordinates are omitted in an observation COPY. Never write to
        # dof_pos/dof_vel: these are views into the simulator state.
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
        collision = torch.stack([(magnitude[:, group] > self.settings.mujica.contact_threshold).any(dim=1)
                                 for group in self.collision_body_groups], dim=1).float()
        states = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)
        wheel_xyz = states[:, self.clearance_body_indices, :3]
        # Vertical clearance from the lower wheel surface to the heightfield.
        # This radius correction is a documented approximation on tilted wheels.
        clearance = (wheel_xyz[..., 2] - self._sample_ground_at(wheel_xyz) -
                     self.settings.mujica.wheel_radius).clamp(min=0.0)
        heights = (self.root_states[:, 2:3] - 0.5 - self.measured_heights).clamp(-1, 1)
        heights = heights * self.obs_scales.height_measurements
        privileged = torch.cat((obs, self.base_lin_vel*self.obs_scales.lin_vel,
                                collision, clearance, heights), dim=-1)
        clip = self.settings.normalization.clip_observations
        return obs.clamp(-clip, clip), privileged.clamp(-clip, clip)

    def compute_observations(self):
        frame, privileged = self._current_frame()
        self.history_buf = torch.cat((frame, self.history_buf[:, :-58]), dim=-1)
        self.privileged_obs_buf = privileged
        if self._pending_reset.any():
            ids = self._pending_reset.nonzero(as_tuple=False).flatten()
            self.history_buf[ids] = frame[ids].repeat(1, 6)
            self._pending_reset[ids] = False

    def get_current_obs(self):
        return self._current_frame()[1]

    def compute_termination_observations(self, env_ids):
        return self._current_frame()[1][env_ids]


    def _reset_wheel_kinematics(self, env_ids):
        states = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)
        # Analytic kinematics for the bundled Go2-W URDF. Fixed hip offsets and
        # link lengths are read once from the asset rather than inferred indices.
        if not hasattr(self, "_fk_description"):
            import os
            import xml.etree.ElementTree as ET
            path = str(self.urdf_path)
            root = ET.parse(os.path.abspath(path)).getroot()
            self._fk_description = {}
            for joint in root.findall("joint"):
                origin = joint.find("origin")
                xyz = [float(x) for x in origin.get("xyz", "0 0 0").split()]
                rpy = [float(x) for x in origin.get("rpy", "0 0 0").split()]
                axis = joint.find("axis")
                self._fk_description[joint.get("name")] = (xyz, rpy,
                    [float(x) for x in axis.get("xyz", "1 0 0").split()] if axis is not None else [1., 0., 0.])
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


    def _reward_dof_vel(self):
        # Same mathematical Go2-W reward, without corrupting live wheel states.
        vel = self.dof_vel.clone()
        vel[:, self.wheel_indices] = 0.0
        return torch.sum(torch.square(vel), dim=1)


    def export_metadata(self):
        def values(tensor):
            return tensor.detach().cpu().flatten().tolist()
        return {
            "joint_names": list(self.dof_names),
            "wheel_indices": values(self.wheel_indices),
            "default_dof_pos": values(self.default_dof_pos),
            "p_gains": values(self.p_gains), "d_gains": values(self.d_gains),
            "torque_limits": values(self.torque_limits),
            "velocity_limits": values(self.dof_vel_limits),
            "action_scale": self.settings.control.action_scale,
            "vel_scale": self.settings.control.vel_scale,
            "clip_actions": self.settings.normalization.clip_actions,
            "control_dt": self.dt, "sim_dt": self.physics_dt,
            "obs_scales": {name: getattr(self.obs_scales, name)
                for name in ("ang_vel", "lin_vel", "dof_pos", "dof_vel")},
            "motor": dict(self.motor_limiter.config),
            "collision_groups": [list(group) for group in COLLISION_GROUPS],
            "clearance_order": ["FL", "FR", "RL", "RR"],
            "wheel_radius": self.settings.mujica.wheel_radius,
            "history_order": "newest_first",
        }

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.settings.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.settings.rewards.tracking_sigma)

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_hip_default(self):
        hip_err = torch.sum((self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]) ** 2, dim = 1)
        # print("penalty",penalty.shape)
        return hip_err

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             3.0 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
