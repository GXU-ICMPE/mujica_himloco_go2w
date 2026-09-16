"""X5loco v8 robot/control/reward defaults adapted to MUJICA's task family.

Copyright 2026 Robot-Nav. Adapted from X5loco; Apache-2.0 (third_party/X5loco-LICENSE).
"""
from dataclasses import asdict
from .robots import X5
from .x5_parameters import X5V8PostureParameters

REWARD_SCALES = dict(
    tracking_lin_vel=2.0, tracking_ang_vel=1.0, lateral_progress=0.75,
    maneuver_completed_step=0.30, joint_pose_corridor=-0.15,
    wheel_position_corridor=-0.25, wheelbase_corridor=-0.15,
    track_width_corridor=-0.15, height_corridor=-0.35, terrain_roll_pitch_l2=-1.0,
    stop_wheel_speed_l2=-0.1, stop_wheel_target_l2=-0.05,
    lin_vel_z=-2.0, ang_vel_xy=-0.05, joint_power=-2e-5,
    action_rate=-0.01, action_smoothness=-0.01, undesired_contacts=-1.0,
    joint_pos_limits=-2.0, leg_joint_acc_l2=-1e-7, wheel_joint_acc_l2=-2e-8,
    leg_joint_torques_l2=-1e-5, wheel_joint_torques_l2=-4e-5,
)


def configure_x5(s):
    s['asset'].update(name='x5', file='resources/robots/x5/urdf/x5.urdf',
                      foot_name='FOOT', wheel_name=['WHEEL'], self_collisions=0,
                      penalize_contacts_on=['thigh', 'calf'])
    s['env'].update(num_envs=512, num_privileged_obs=X5.critic_dim,
                    num_one_step_privileged_obs=X5.critic_dim)
    s['init_state'].update(pos=[0., 0., 0.587], default_joint_angles={
        n: (0.8 if n.endswith('HFE') else -1.5 if n.endswith('KFE') else 0.) for n in X5.joints})
    s['control'].update(action_scale=0.2, vel_scale=1 / 0.1005, decimation=8,
                        stiffness=dict(HAA=800., HFE=800., KFE=800., WHEEL=0.),
                        damping=dict(HAA=20., HFE=20., KFE=20., WHEEL=3.),
                        wheel_target_limit=24., position_target_limits={
                            n: ([-0.79, 0.64] if n.startswith(('RF', 'RH')) else [-0.64, 0.79])
                            if n.endswith('HAA') else [-1.60, 2.39] if n.endswith('HFE') else [-2.32, -0.52]
                            for n in X5.joints[:12]},
                        stop_pi=dict(enabled=True, enter_linear=0.02, exit_linear=0.03,
                                     enter_yaw=0.04, exit_yaw=0.05, dwell_s=0.2, ramp_s=0.2,
                                     contact_dwell_s=0.02, contact_force_n=5., ki=3.,
                                     integral_limit=12., torque_limit=30.))
    s['mujica'].update(wheel_radius=0.1005, contact_threshold=5., termination_contact_threshold=10.,
                       height_observation_offset=0.567)
    # The bundled Go2 speed/torque envelope has no X5 calibration.
    s['mujica']['motor']['enabled'] = False
    s['sim']['dt'] = 0.0025
    s['sim']['physx'].update(num_position_iterations=8, num_velocity_iterations=4,
                            contact_offset=0.002, rest_offset=0., max_gpu_contact_pairs=1048576)
    s['domain_rand'].update(
        delay=False, disturbance=False, randomize_motor_strength=False,
        payload_mass_range=[-1., 1.], randomize_link_mass=True, link_mass_range=[0.95, 1.05],
        com_displacement_range=[-0.01, 0.01], kp_range=[0.95, 1.05], kd_range=[0.95, 1.05],
        friction_range=[0.6, 1.2], randomize_restitution=True, restitution_range=[0., 0.05],
        initial_joint_pos_range=[-0.05, 0.05], motor_offset_range=[-0.005, 0.005],
        randomize_motor_offset=True, max_push_vel_xy=0.15)
    # Keep MUJICA's terrain curriculum. V8 pure-axis/stop coverage uses fixed
    # conservative initial ranges; the X5loco per-axis EMA curriculum is separate.
    s['commands']['ranges'].update(lin_vel_x=[-0.5, 0.5], lin_vel_y=[-0.2, 0.2], ang_vel_yaw=[-0.5, 0.5])
    s['commands']['sampling'] = dict(profile='x5_v8_coverage', stop=0.10, x=0.15, y=0.25, yaw=0.10)
    s['rewards'] = dict(profile='x5_v8_mujica_v1', scales=REWARD_SCALES.copy(),
        base_height_target=0.567, only_positive_rewards=False, soft_dof_pos_limit=0.9,
        soft_dof_vel_limit=1., soft_torque_limit=1., tracking_sigma=0.16,
        posture=asdict(X5V8PostureParameters()),
        terrain_geometry=dict(slope_transition_deg=[3., 8.], residual_transition_m=[0.008, 0.030],
            jump_transition_m=[0.020, 0.050], lateral_transition=[0.05, 0.20], yaw_transition=[0.10, 0.40],
            stop_linear_threshold=0.03, stop_yaw_threshold=0.05, flat_tilt_allowance_deg=3.,
            slope_tilt_margin_deg=5., slope_tilt_cap_deg=30., rough_tilt_allowance_deg=15.,
            nonflat_tilt_scale=0.5, invalid_tilt_allowance_deg=10.),
        lin_vel_z_curriculum=dict(start_iteration=0, end_iteration=1500, final_weight=0.))
