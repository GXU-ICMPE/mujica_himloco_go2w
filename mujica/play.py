"""Deterministic evaluation for either stage and either simulator backend."""
import torch
from .skills import SKILL_NAMES


@torch.no_grad()
def play(runner, steps=2000, skill="auto"):
    env = runner.env
    if skill == "auto" and runner.stage != "s2":
        raise ValueError("Automatic selection requires S2; choose --skill " + "/".join(SKILL_NAMES) + " for S1")
    env.reset()
    hidden = runner.low_model.initial_hidden(env.num_envs, runner.device)
    runner.low_model.eval()
    runner.model.eval()
    totals = torch.zeros(env.num_envs, device=runner.device)
    counts = torch.zeros(3, device=runner.device)
    for _ in range(steps):
        obs, critic = runner._observations()
        if skill == "auto":
            chosen = runner.model.act(runner.selector_history(obs), critic, deterministic=True)["actions"]
        else:
            chosen = torch.full((env.num_envs,), SKILL_NAMES.index(skill),
                                device=runner.device, dtype=torch.long)
        env.set_skill(chosen.to(env.device))
        obs = env.get_observations().to(runner.device)
        actions, next_hidden = runner.low_model.act_inference(obs, hidden)
        _, _, reward, done, _, _, _ = env.step(actions.to(env.device))
        hidden = runner.low_model.reset_hidden(next_hidden, done.to(runner.device))
        totals += reward.to(runner.device)
        counts += torch.bincount(chosen, minlength=3)
    print({"steps": steps, "mean_accumulated_reward": totals.mean().item(),
           "skill_fractions": (counts / counts.sum()).cpu().tolist()})
