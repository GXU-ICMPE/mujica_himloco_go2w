"""Explicit engineering defaults; PPO values inherit GO2WRoughCfgPPO."""
import copy

MODEL_DEFAULTS = dict(frame_dim=58, history_len=6, critic_dim=270,
                      action_dim=16, collision_dim=18, latent_dim=16)
SELECTOR_DEFAULTS = dict(history_len=6, base_frame_dim=57, critic_dim=270, skill_count=3)
PPO_DEFAULTS = dict(num_learning_epochs=5, num_mini_batches=4, clip_param=0.2,
                    gamma=0.99, lam=0.95, value_loss_coef=1.0,
                    entropy_coef=0.005, learning_rate=1e-3, max_grad_norm=1.0,
                    use_clipped_value_loss=True, schedule="adaptive", desired_kl=0.01)


def training_config(stage="s1"):
    if stage not in ("s1", "s2"):
        raise ValueError("stage must be s1 or s2")
    return copy.deepcopy(dict(stage=stage, model=MODEL_DEFAULTS, selector=SELECTOR_DEFAULTS,
                              ppo=PPO_DEFAULTS, steps_per_env=48,
                              iterations=30000 if stage == "s1" else 10000,
                              save_interval=1000, seed=1))
