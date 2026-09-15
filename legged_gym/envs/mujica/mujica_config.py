"""MUJICA architecture on upstream HIMLoco's Go2-W task.

S1 shares the Go2-W reward terms, with softer posture penalties on discrete
obstacles and stairs. Positive clipping and S2 velocity tracking are retained.
"""
from legged_gym.envs.go2w.go2w_config import GO2WRoughCfg, GO2WRoughCfgPPO


class MUJICACfg(GO2WRoughCfg):
    class env(GO2WRoughCfg.env):
        num_one_step_observations = 58
        num_observations = 58 * 6
        num_one_step_privileged_obs = 270
        num_privileged_obs = 270
        episode_length_s = 20.0

    class mujica:
        stage = "s1"
        task_family = "terrain_locomotion"
        task_contract_version = 2
        terrain_edge_margin = 0.5
        curriculum_min_distance = 1.5
        contact_threshold = 1.0
        wheel_radius = 0.086
        # Curriculum thresholds are explicit engineering choices: not published.
        tracking_success_fraction = 0.8
        tracking_success_seconds = 3.0
        distance_failure_fraction = 0.5
        motor = {
            "enabled": True,
            "calibrated": False,
            # Fig.3 suggests the knee is near half the no-load speed.
            "knee_fraction": 0.5,
            "calf_min_factor": 0.3,
            "calf_cos_offset": 0.0,
            "calf_cos_amplitude": 1.0,
            # Fig.3 peaks near -pi/2. Coefficients remain uncalibrated.
            "calf_cos_phase": -1.5707963267948966,
        }

    class terrain(GO2WRoughCfg.terrain):
        # Six physical terrain variants; task allocation is balanced separately.
        num_rows = 20
        num_cols = 18
        max_init_terrain_level = 0
        terrain_length = 8.0
        terrain_width = 8.0
        horizontal_scale = 0.1
        border_size = 10.0
        curriculum = True
        measure_heights = True
        platform_size = 2.0
        slope_range = [0.0, 0.3]
        stair_width = 0.3
        stair_height_range = [0.05, 0.23]
        discrete_size_range = [0.25, 0.75]
        discrete_height_range = [0.05, 0.22]
        discrete_num_obstacles = 40

    class commands(GO2WRoughCfg.commands):
        # Terrain progression is enabled; upstream's env-index-based velocity
        # curriculum would introduce unintended correlations with skill groups.
        curriculum = False
        heading_command = False
        resampling_time = 10.0

    class rewards(GO2WRoughCfg.rewards):
        class terrain_adaptation:
            version = 1
            complex_scales = {
                "orientation": -0.35,
                "base_height": -7.0,
                "run_still": -0.025,
                "stand_still": -0.35,
            }
            stand_still_linear_threshold = 0.1
            stand_still_yaw_threshold = 0.1
            complex_stand_still_joint_tolerance = 0.1
            base_height_patch_half_length = 0.4
            base_height_patch_half_width = 0.3
            complex_base_height_tolerance = 0.03

    class control(GO2WRoughCfg.control):
        decimation = 4

    class sim(GO2WRoughCfg.sim):
        dt = 0.005


class MUJICAS2Cfg(MUJICACfg):
    class mujica(MUJICACfg.mujica):
        stage = "s2"


class MUJICACfgPPO(GO2WRoughCfgPPO):
    """Registry metadata; train via python -m mujica.train (dedicated runner)."""
    class runner(GO2WRoughCfgPPO.runner):
        experiment_name = "MUJICA_GO2W"
        max_iterations = 30000
