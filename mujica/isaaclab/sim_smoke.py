"""Opt-in Isaac Lab physics smoke with zero actions and no PPO updates.

This checks scene creation and finite tensors; it does not validate locomotion.
"""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=9)
    parser.add_argument("--steps", type=int, default=350)
    parser.add_argument("--stage", choices=("s1", "s2"), default="s1")
    parser.add_argument("--terrain", choices=("plane", "trimesh"), default="trimesh")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 1:
        raise SystemExit("num-envs and steps must be positive")
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(device=args.device, headless=args.headless)
    env = None
    try:
        import torch
        from .adapter import MUJICAVecEnv
        from .env import MUJICAEnv
        from .env_cfg import build_env_cfg
        from .settings import default_settings
        settings = default_settings()
        settings.env.num_envs = args.num_envs
        settings.mujica.stage = args.stage
        settings.terrain.mesh_type = args.terrain
        settings.terrain.num_rows, settings.terrain.num_cols = 2, 6
        env = MUJICAVecEnv(MUJICAEnv(build_env_cfg(settings, args.device, seed=1)))
        obs, critic = env.reset()
        assert obs.shape == (args.num_envs, 348) and critic.shape == (args.num_envs, 270)
        action = torch.zeros(args.num_envs, 16, device=env.device)
        done_count = 0
        for step in range(args.steps):
            if args.stage == "s2":
                env.set_skill(torch.full((args.num_envs,), (step // 50) % 3, device=env.device))
            obs, critic, reward, dones, info, ids, terminal = env.step(action)
            for name, value in (("policy", obs), ("critic", critic), ("reward", reward), ("terminal", terminal),
                                ("root", env.env.root_states), ("q", env.env.dof_pos), ("qd", env.env.dof_vel),
                                ("torque", env.env.torques), ("contacts", env.env.contact_forces)):
                if not torch.isfinite(value).all():
                    raise RuntimeError(f"Non-finite {name} at step {step}")
            if not torch.equal(ids, dones.nonzero().flatten()):
                raise RuntimeError("Terminal IDs do not match reset flags")
            if terminal.shape != (len(ids), 270):
                raise RuntimeError("Invalid terminal critic shape")
            done_count += int(dones.sum())
        print(json.dumps(dict(status="PASS", verification="physics_interface_only", stage=args.stage,
            steps=args.steps, resets=done_count, metadata=env.export_metadata()), indent=2))
    finally:
        if env is not None:
            env.close()
        launcher.app.close()


if __name__ == "__main__":
    main()
