"""Train MUJICA S1 or S2; native Isaac Lab is the default simulator."""
import argparse
import json
import random
from pathlib import Path
from .skills import SKILL_NAMES


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("isaaclab", "isaacgym"), default="isaaclab")
    p.add_argument("--robot", choices=("x5", "go2w"), default=None,
                   help="New Isaac Lab runs default to x5; resume/S2 inherit checkpoint robot")
    p.add_argument("--terrain", choices=("trimesh", "plane"), default=None)
    p.add_argument("--terrain-rows", type=int, default=None)
    p.add_argument("--terrain-cols", type=int, default=None)
    p.add_argument("--no-randomization", action="store_true", help="Disable domain randomization for diagnostics")
    p.add_argument("--check-config", action="store_true", help="Offline asset/settings validation, without launching Isaac Sim")
    p.add_argument("--stage", choices=("s1", "s2"), default="s1")
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--iterations", type=int, help="Additional training iterations")
    p.add_argument("--steps-per-env", type=int)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--log-dir", help="Defaults to logs/mujica_lab_<stage>/<timestamp>")
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


def prepare_training(args):
    """Validate arguments, checkpoints and settings before starting the simulator."""
    import copy
    from .config import training_config
    from .runner import load_checkpoint
    from .isaaclab.settings import default_settings, settings_from_dict, validate_settings, validate_checkpoint_backend
    from .isaaclab.robots import robot_spec

    config, saved = training_config(args.stage), None
    if args.resume:
        saved = load_checkpoint(args.resume)
        if saved["config"]["stage"] != args.stage:
            raise ValueError("--stage must match --resume checkpoint")
        validate_checkpoint_backend(saved, resume=True)
        config = copy.deepcopy(saved["config"])
    elif args.low_level:
        saved = load_checkpoint(args.low_level)
        validate_checkpoint_backend(saved, resume=False)
        config["model"] = copy.deepcopy(saved["config"]["model"])
    if args.stage == "s2" and not (args.resume or args.low_level):
        raise ValueError("S2 requires --low-level or an S2 --resume checkpoint")
    if args.play and not args.resume:
        raise ValueError("--play requires --resume")
    if args.play and args.stage == "s1" and args.skill == "auto":
        raise ValueError("S1 playback requires --skill " + "/".join(SKILL_NAMES))
    if args.low_level and (args.stage != "s2" or args.resume):
        raise ValueError("--low-level is only used to start a new S2 run")
    snapshot = (saved or {}).get("metadata", {}).get("environment_config")
    saved_robot = (snapshot or {}).get('asset', {}).get('name', 'go2w')
    robot = getattr(args, 'robot', None) or (saved_robot if saved else 'x5')
    if saved and robot != saved_robot:
        raise ValueError(f"Checkpoint is for {saved_robot}, requested {robot}; start a fresh S1 run")
    spec = robot_spec(robot)
    if not args.resume:
        config['model'].update(critic_dim=spec.critic_dim, collision_dim=len(spec.collision_groups))
        config['selector']['critic_dim'] = spec.critic_dim
    if args.config:
        override = json.loads(Path(args.config).read_text())
        allowed = {"model", "selector", "ppo", "steps_per_env", "save_interval"}
        if set(override) - allowed:
            raise ValueError("Unsupported config keys: " + str(set(override) - allowed))
        for key, value in override.items():
            if isinstance(config[key], dict):
                config[key].update(value)
            else:
                config[key] = value
    if args.steps_per_env is not None:
        config["steps_per_env"] = args.steps_per_env
    if config["steps_per_env"] < 1 or config["save_interval"] < 1:
        raise ValueError("steps-per-env and save_interval must be positive")
    if args.resume and config != saved["config"]:
        raise ValueError("Resume configuration differs: use the checkpoint algorithm configuration")
    if not args.resume:
        config["seed"] = args.seed
    if (config['model']['critic_dim'] != spec.critic_dim
            or config['model']['collision_dim'] != len(spec.collision_groups)
            or config['selector']['critic_dim'] != spec.critic_dim):
        raise ValueError(f"Model dimensions must match the {robot} observation contract")
    settings = settings_from_dict(copy.deepcopy(snapshot)) if snapshot else default_settings(robot)
    settings.mujica.stage = args.stage
    settings.env.num_envs = args.num_envs
    before = json.dumps(settings, sort_keys=True)
    if args.no_noise:
        settings.noise.add_noise = False
    if args.motor_config:
        settings.mujica.motor.update(json.loads(Path(args.motor_config).read_text()))
    if args.disable_motor_model:
        settings.mujica.motor.enabled = False
    if args.terrain:
        settings.terrain.mesh_type = args.terrain
    if args.terrain_rows is not None:
        settings.terrain.num_rows = args.terrain_rows
    if args.terrain_cols is not None:
        settings.terrain.num_cols = args.terrain_cols
    if args.no_randomization:
        for name, value in settings.domain_rand.items():
            if isinstance(value, bool):
                settings.domain_rand[name] = False
    if args.resume and not args.play and before != json.dumps(settings, sort_keys=True):
        raise ValueError("Training resume must preserve saved environment settings; use a new run for a changed task")
    validate_settings(settings)
    return config, settings


def main():
    args = parser().parse_args()
    if args.num_envs < 1 or args.play_steps < 1 or (args.iterations is not None and args.iterations < 1):
        raise SystemExit("num-envs, play-steps and iterations must be positive")
    if args.backend == "isaacgym":
        if args.robot == 'x5':
            raise SystemExit('X5 is supported by the Isaac Lab backend only')
        if any((args.terrain, args.terrain_rows, args.terrain_cols, args.no_randomization, args.check_config)):
            raise SystemExit("Terrain/diagnostic flags in this entrypoint require --backend isaaclab")
        # Legacy implementation owns the mandatory isaacgym-before-torch import.
        from .train_isaacgym import main as train_gym
        train_gym(args)
        return
    try:
        config, settings = prepare_training(args)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
    if args.check_config:
        from .isaaclab.check import check
        print(json.dumps(check(settings), indent=2, ensure_ascii=False))
        return
    # AppLauncher must precede env_cfg/env imports (pxr/omni/PhysX initialization).
    try:
        from isaaclab.app import AppLauncher
    except ImportError as exc:
        raise SystemExit("Use an Isaac Lab Python environment, e.g. x3w_isaaclab. Offline checks: --check-config") from exc
    launcher = AppLauncher(headless=args.headless, device=args.device)
    env = None
    try:
        import numpy as np
        import torch
        from datetime import datetime
        from .isaaclab.env_cfg import build_env_cfg
        from .isaaclab.env import MUJICAEnv
        from .isaaclab.adapter import MUJICAVecEnv
        from .runner import MUJICARunner

        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        env = MUJICAVecEnv(MUJICAEnv(build_env_cfg(settings, args.device, config["seed"])))
        log_dir = args.log_dir or str(Path("logs") / ("mujica_lab_" + settings.asset.name + '_' + args.stage) /
                                     datetime.now().strftime("%Y%m%d_%H%M%S"))
        low_level = args.low_level or (args.resume if args.stage == "s2" else None)
        runner = MUJICARunner(env, config, log_dir, device=args.device, low_level=low_level)
        if args.resume:
            runner.load(args.resume, load_optimizer=not args.play)
            env.set_training_iteration(runner.iteration)
        if args.play:
            from .play import play
            play(runner, steps=args.play_steps, skill=args.skill)
        else:
            runner.learn(iterations=args.iterations)
    finally:
        if env is not None:
            env.close()
        launcher.app.close()


if __name__ == "__main__":
    main()
