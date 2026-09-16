# Copyright 2026 Robot-Nav. Apache-2.0; see third_party/X5loco-LICENSE.
# Vendored from X5loco v8; used by the MUJICA adapter.
"""V8 references and physical deadbands (metres, radians, seconds)."""

from dataclasses import dataclass


@dataclass
class X5V8PostureParameters:
    # sevnce_motion/config/x5/robot.info: standPose, not defaultJointState.
    stand_joint_angles: tuple[float, float, float] = (0.0, 0.80, -1.50)
    # URDF axle-origin FK, RF/LF/RH/LH. Wheel radius = 0.1005 m.
    reference_wheel_xy_m: tuple[tuple[float, float], ...] = (
        (0.354674, -0.326500), (0.355118, 0.326500),
        (-0.401926, -0.326500), (-0.401482, 0.326500),
    )
    reference_base_height_m: float = 0.567
    stop_joint_deadband: tuple[float, float, float] = (0.04, 0.06, 0.08)
    rolling_joint_deadband: tuple[float, float, float] = (0.12, 0.30, 0.35)
    maneuver_joint_deadband: tuple[float, float, float] = (0.25, 0.50, 0.60)
    nonflat_extra_joint_deadband: tuple[float, float, float] = (0.12, 0.25, 0.35)
    joint_error_scale: tuple[float, float, float] = (0.15, 0.30, 0.30)
    stop_xy_deadband_m: tuple[float, float] = (0.035, 0.025)
    rolling_xy_deadband_m: tuple[float, float] = (0.055, 0.045)
    maneuver_extra_xy_deadband_m: tuple[float, float] = (0.065, 0.065)
    nonflat_extra_xy_deadband_m: tuple[float, float] = (0.070, 0.070)
    xy_error_scale_m: float = 0.06
    wheelbase_shortening_m: float = 0.07
    track_narrowing_m: float = 0.06
    span_error_scale_m: float = 0.06
    stop_height_deadband_m: float = 0.015
    rolling_height_deadband_m: float = 0.030
    maneuver_extra_height_deadband_m: float = 0.020
    nonflat_extra_height_deadband_m: float = 0.080
    height_error_scale_m: float = 0.05
    recenter_time_s: float = 0.5
    stop_tighten_time_s: float = 0.4
    contact_threshold_n: float = 5.0
    support_vertical_force_n: float = 20.0
    contact_confirmation_s: float = 0.04
    step_air_time_range_s: tuple[float, float] = (0.10, 0.70)
    step_min_clearance_m: float = 0.025
    step_target_clearance_m: float = 0.06
    scan_nearest_radius_m: float = 0.15
