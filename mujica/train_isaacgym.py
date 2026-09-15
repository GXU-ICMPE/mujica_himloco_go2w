"""Train MUJICA S1 or S2 on the bundled Isaac Gym Go2W environment."""
import argparse
import json
import random
from pathlib import Path
from .skills import SKILL_NAMES, validate_skill_metadata


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("s1", "s2"), default="s1")
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--iterations", type=int, help="Additional training iterations")
    p.add_argument("--steps-per-env", type=int)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--log-dir", help="Defaults to logs/mujica_<stage>/<timestamp>")
    p.add_argument("--low-level", help="S1 checkpoint required for S2")
    p.add_argument("--resume", help="Continue optimization from a checkpoint of the same stage")
    p.add_argument("--config", help="JSON overrides: model, selector, ppo, steps_per_env, save_interval")
    p.add_argument("--motor-config", help="JSON motor envelope override, e.g. measured calibration")
    p.add_argument("--disable-motor-model", action="store_true", help="Use static URDF torque limits")
    p.add_argument("--no-noise", action="store_true", default=None)
    p.add_argument("--play", action="store_true", help="Evaluate --resume without any updates")
    p.add_argument("--play-steps", type=int, default=2000)
    p.add_argument("--skill", choices=("auto", *SKILL_NAMES), default="auto")
    return p


def main(args=None):
    args = parser().parse_args() if args is None else args
    if args.num_envs < 1 or args.play_steps < 1:
        raise SystemExit("num-envs and play-steps must be positive")
    # This ordering is a requirement of the upstream Isaac Gym binary.
    try:
        import isaacgym  # noqa: F401
        from isaacgym import gymapi, gymutil
    except ImportError as exc:
        raise SystemExit("Isaac Gym Preview 4 is required for simulator training. Install its python/ "
                         "package in the HIMLoco environment first. CPU algorithm tests: "
                         "python -m mujica.smoke") from exc
    import numpy as np
    import torch
    from datetime import datetime
    from legged_gym.envs.mujica.mujica_config import MUJICACfg
    from legged_gym.envs.mujica.mujica_robot import MUJICARobot
    from legged_gym.utils.helpers import class_to_dict
    from .config import training_config
    from .runner import MUJICARunner, load_checkpoint
    from .rewards import validate_reward_settings

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = training_config(args.stage)
    saved = None
    if args.resume:
        saved = load_checkpoint(args.resume)
        validate_skill_metadata(saved.get("metadata", {}))
        config = saved["config"]
        if config["stage"] != args.stage:
            raise SystemExit("--stage must match --resume checkpoint")
    elif args.low_level:
        saved = load_checkpoint(args.low_level)
        validate_skill_metadata(saved.get("metadata", {}))
        config["model"] = saved["config"]["model"]
    if args.config:
        override = json.loads(Path(args.config).read_text())
        allowed = {"model", "selector", "ppo", "steps_per_env", "save_interval"}
        if set(override) - allowed:
            raise SystemExit("Unsupported config keys: " + str(set(override) - allowed))
        for key, value in override.items():
            if isinstance(config[key], dict):
                config[key].update(value)
            else:
                config[key] = value
    if args.steps_per_env is not None:
        config["steps_per_env"] = args.steps_per_env
    if not args.resume:
        config["seed"] = args.seed
    # Recreate the terrain from the saved run seed before constructing PhysX.
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    cfg = MUJICACfg()
    # Resume both controller and physical settings. Reconstructing default
    # motor/noise settings would silently change the learned control problem.
    def apply_snapshot(target, values):
        for name, value in values.items():
            if not hasattr(target, name):
                raise ValueError("Unknown saved environment setting: " + name)
            current = getattr(target, name)
            if isinstance(value, dict) and not isinstance(current, dict):
                apply_snapshot(current, value)
            else:
                setattr(target, name, value)
    snapshot = (saved or {}).get("metadata", {}).get("environment_config")
    if snapshot:
        validate_reward_settings(snapshot["rewards"], snapshot["terrain"])
        apply_snapshot(cfg, snapshot)
    elif saved and "motor" in saved.get("metadata", {}):
        cfg.mujica.motor = saved["metadata"]["motor"].copy()
    cfg.mujica.stage = args.stage
    cfg.env.num_envs = args.num_envs
    old_motor = json.dumps(cfg.mujica.motor, sort_keys=True)
    old_noise = cfg.noise.add_noise
    if args.no_noise is not None:
        cfg.noise.add_noise = False
    if args.motor_config:
        cfg.mujica.motor.update(json.loads(Path(args.motor_config).read_text()))
    if args.disable_motor_model:
        cfg.mujica.motor["enabled"] = False
    if args.resume and not args.play and (old_motor != json.dumps(cfg.mujica.motor, sort_keys=True)
                                         or old_noise != cfg.noise.add_noise):
        raise SystemExit("Training resume must preserve saved motor/noise settings; start a new run "
                         "for a changed environment. --play allows explicit evaluation overrides.")
    # Capture before Isaac converts interval seconds into step counts.
    environment_snapshot = class_to_dict(cfg)
    sim = gymapi.SimParams()
    gymutil.parse_sim_config(class_to_dict(cfg.sim), sim)
    sim.use_gpu_pipeline = args.device.startswith("cuda")
    sim.physx.use_gpu = args.device.startswith("cuda")
    env = MUJICARobot(cfg, sim, gymapi.SIM_PHYSX, args.device, args.headless)
    env._mujica_config_snapshot = environment_snapshot
    log_dir = args.log_dir or str(Path("logs") / ("mujica_" + args.stage) /
                                  datetime.now().strftime("%Y%m%d_%H%M%S"))
    low_level = args.low_level or (args.resume if args.stage == "s2" else None)
    runner = MUJICARunner(env, config, log_dir, device=args.device, low_level=low_level)
    if args.resume:
        runner.load(args.resume, load_optimizer=not args.play)
    if args.play:
        if not args.resume:
            raise SystemExit("--play requires --resume checkpoint")
        from .play import play
        play(runner, steps=args.play_steps, skill=args.skill)
    else:
        runner.learn(iterations=args.iterations)


if __name__ == "__main__":
    main()
