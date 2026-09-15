"""Two-stage on-policy training with recurrent-state and terminal bookkeeping."""
import copy
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from .models import MultiSkillActorCritic, SelectorActorCritic
from .ppo import PPO
from .storage import RolloutStorage
from .skills import SKILL_NAMES, validate_skill_metadata


def load_checkpoint(path, device="cpu"):
    # Only load your own trusted training checkpoints (optimizer/RNG state).
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch 1.x used with Isaac Gym
        return torch.load(path, map_location=device)


class MUJICARunner:
    def __init__(self, env, config, log_dir, device="cpu", low_level=None):
        self.env, self.device = env, torch.device(device)
        self.config = copy.deepcopy(config)
        self.stage = config["stage"]
        if self.stage not in ("s1", "s2"):
            raise ValueError("Only S1 and S2 exist in MUJICA")
        self.metadata = env.export_metadata() if hasattr(env, "export_metadata") else {}
        validate_skill_metadata(self.metadata)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.iteration = 0
        self.total_steps = 0
        self.low_model = MultiSkillActorCritic(**config["model"]).to(self.device)
        if self.stage == "s2":
            if not low_level:
                raise ValueError("S2 requires a trained S1 checkpoint via --low-level")
            ckpt = load_checkpoint(low_level, self.device)
            validate_skill_metadata(ckpt.get("metadata", {}))
            if ckpt["config"]["model"] != config["model"]:
                raise ValueError("Low-level model configuration must match the S1 checkpoint")
            self.low_model.load_state_dict(ckpt["low_model"])
            self.low_model.requires_grad_(False)
            self.low_model.eval()
            self.model = SelectorActorCritic(**config["selector"]).to(self.device)
        else:
            self.model = self.low_model
        self.algorithm = PPO(self.model, **config["ppo"])
        self.storage = RolloutStorage(env.num_envs, config["steps_per_env"], device=self.device)
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(str(self.log_dir))
        except ImportError:
            pass  # metrics.jsonl is always written
        if hasattr(env, "_mujica_config_snapshot"):
            self.metadata["environment_config"] = env._mujica_config_snapshot
        self.metadata.update(frame_dim=58, history_len=6)
        self._write_json("config.json", self.config)
        self._write_json("environment.json", self.metadata)

    def _write_json(self, filename, value):
        (self.log_dir / filename).write_text(json.dumps(value, indent=2), encoding="utf-8")

    def _observations(self):
        return (self.env.get_observations().to(self.device),
                self.env.get_privileged_observations().to(self.device))

    @staticmethod
    def selector_history(history):
        return history.reshape(history.shape[0], 6, 58)[..., :57].reshape(history.shape[0], -1)

    def learn(self, iterations=None):
        iterations = self.config["iterations"] if iterations is None else iterations
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        self.env.reset()
        obs, critic = self._observations()
        hidden = self.low_model.initial_hidden(self.env.num_envs, device=self.device)
        episode_return = torch.zeros(self.env.num_envs, device=self.device)
        recent_returns = []
        self.model.train()
        stop_at = self.iteration + iterations
        while self.iteration < stop_at:
            start = time.perf_counter()
            episode_metrics = {}
            episode_counts = {}
            skill_counts = torch.zeros(3, device=self.device)
            motor_violations = []
            with torch.no_grad():
                for _ in range(self.config["steps_per_env"]):
                    history_before, critic_before, hidden_before = obs.clone(), critic.clone(), hidden.clone()
                    if self.stage == "s1":
                        result = self.model.act(history_before, critic_before, hidden_before)
                        motor_actions = result["actions"]
                        hidden_next = result["hidden_next"]
                    else:
                        selector_obs = self.selector_history(history_before)
                        result = self.model.act(selector_obs, critic_before)
                        self.env.set_skill(result["actions"].to(self.env.device))
                        conditioned_obs = self.env.get_observations().to(self.device)
                        low = self.low_model.act(conditioned_obs, critic_before, hidden_before, deterministic=True)
                        motor_actions, hidden_next = low["actions"], low["hidden_next"]
                    out = self.env.step(motor_actions.to(self.env.device))
                    next_obs, next_critic, rewards, dones, info, terminal_ids, terminal_critic = out
                    obs, critic = next_obs.to(self.device), next_critic.to(self.device)
                    rewards = rewards.to(self.device).clone()
                    dones = dones.to(self.device).bool().clone()
                    # Auto-reset observations cannot supervise a predecessor or
                    # bootstrap its timeout: restore the actual terminal state.
                    successor_critic = critic.clone()
                    terminal_ids = terminal_ids.to(self.device).long()
                    successor_critic[terminal_ids] = terminal_critic.to(self.device)
                    timeouts = info.get("time_outs", torch.zeros_like(dones)).to(self.device).bool()
                    timeout_values = self.model.critic(successor_critic).squeeze(-1)
                    bootstrapped_rewards = rewards + self.config["ppo"]["gamma"] * timeout_values * timeouts
                    fields = dict(policy_inputs=result["policy_inputs"], critic_obs=critic_before,
                                  actions=result["actions"], values=result["values"],
                                  log_probs=result["log_probs"], rewards=bootstrapped_rewards, dones=dones)
                    if self.stage == "s1":
                        fields.update(old_mean=result["old_mean"], old_std=result["old_std"],
                                      history=history_before, hidden=hidden_before,
                                      velocity_targets=critic_before[:, 58:61],
                                      collision_targets=critic_before[:, 61:79],
                                      wheel_targets=critic_before[:, 79:83],
                                      successor_obs=successor_critic[:, :58])
                    else:
                        fields["old_logits"] = result["old_logits"]
                    self.storage.add(**fields)
                    if self.stage == "s2":
                        skill_counts += torch.bincount(result["actions"].long(), minlength=3)
                    for name, value in info.get("episode", {}).items():
                        scalar = float(torch.as_tensor(value).float().mean())
                        weight = 1.0
                        if name.startswith(("task/", "terrain/")) and not name.endswith("/episodes"):
                            count_key = "/".join(name.split("/")[:2]) + "/episodes"
                            weight = float(info["episode"].get(count_key, 1.0))
                        episode_metrics[name] = episode_metrics.get(name, 0.0) + scalar * weight
                        episode_counts[name] = episode_counts.get(name, 0.0) + weight
                    if "motor_violation_count" in info:
                        motor_violations.append(float(info["motor_violation_count"].float().mean()))
                    hidden = self.low_model.reset_hidden(hidden_next, dones)
                    episode_return += rewards
                    if dones.any():
                        recent_returns.extend(episode_return[dones].cpu().tolist())
                        recent_returns = recent_returns[-100:]
                        episode_return[dones] = 0
                last_values = self.model.critic(critic).squeeze(-1)
                self.storage.compute_returns(last_values, gamma=self.config["ppo"]["gamma"],
                                             lam=self.config["ppo"]["lam"])
            collection_seconds = time.perf_counter() - start
            update_start = time.perf_counter()
            metrics = self.algorithm.update(self.storage)
            self.storage.clear()
            self.iteration += 1
            self.total_steps += self.env.num_envs * self.config["steps_per_env"]
            metrics.update(iteration=self.iteration, total_steps=self.total_steps,
                           collection_seconds=collection_seconds,
                           update_seconds=time.perf_counter() - update_start)
            if recent_returns:
                metrics["episode_return_mean"] = float(np.mean(recent_returns))
            metrics.update({"episode/" + key: value if key.endswith("/episodes") else value / max(episode_counts[key], 1.0)
                            for key, value in episode_metrics.items()})
            if motor_violations:
                metrics["motor_requested_violations_mean"] = float(np.mean(motor_violations))
            if self.stage == "s2":
                fractions = skill_counts / skill_counts.sum().clamp(min=1)
                for i, name in enumerate(SKILL_NAMES):
                    metrics["selector_fraction/" + name] = fractions[i].item()
            with (self.log_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(metrics) + "\n")
            if self.writer:
                for name, value in metrics.items():
                    self.writer.add_scalar(name, value, self.iteration)
            if self.iteration == 1 or self.iteration % 10 == 0 or self.iteration == stop_at:
                print(json.dumps(metrics), flush=True)
            if self.iteration % self.config["save_interval"] == 0:
                self.save(self.log_dir / ("model_%06d.pt" % self.iteration))
        self.save(self.log_dir / ("model_%06d.pt" % self.iteration))
        self.save(self.log_dir / "last.pt")
        if self.writer:
            self.writer.flush()

    def save(self, path):
        payload = dict(format_version=1, config=self.config, metadata=self.metadata,
                       iteration=self.iteration, total_steps=self.total_steps,
                       low_model=self.low_model.state_dict(),
                       selector=self.model.state_dict() if self.stage == "s2" else None,
                       optimizer=self.algorithm.optimizer.state_dict(),
                       rng_torch=torch.get_rng_state(), rng_numpy=np.random.get_state(),
                       rng_python=random.getstate())
        estimator_opt = getattr(self.algorithm, "estimator_optimizer", None)
        payload["estimator_optimizer"] = estimator_opt.state_dict() if estimator_opt else None
        payload["learning_rate"] = self.algorithm.learning_rate
        if torch.cuda.is_available():
            payload["rng_cuda"] = torch.cuda.get_rng_state_all()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(str(temporary), str(path))

    def load(self, path, load_optimizer=True):
        payload = load_checkpoint(path, self.device)
        validate_skill_metadata(payload.get("metadata", {}))
        if payload["config"] != self.config:
            raise ValueError("Resume configuration differs: use checkpoint config or start a new run")
        if load_optimizer and payload.get("metadata", {}).get("reward_profile") != self.metadata.get("reward_profile"):
            raise ValueError("Resume reward profile differs: restore checkpoint reward settings or start a new S1 run")
        self.low_model.load_state_dict(payload["low_model"])
        if self.stage == "s2":
            self.model.load_state_dict(payload["selector"])
        if load_optimizer:
            self.algorithm.optimizer.load_state_dict(payload["optimizer"])
            est_opt = getattr(self.algorithm, "estimator_optimizer", None)
            if est_opt and payload.get("estimator_optimizer"):
                est_opt.load_state_dict(payload["estimator_optimizer"])
            self.algorithm.learning_rate = payload["learning_rate"]
        self.iteration, self.total_steps = payload["iteration"], payload["total_steps"]
        torch.set_rng_state(payload["rng_torch"].cpu())
        np.random.set_state(payload["rng_numpy"])
        random.setstate(payload["rng_python"])
        if torch.cuda.is_available() and "rng_cuda" in payload:
            torch.cuda.set_rng_state_all([state.cpu() for state in payload["rng_cuda"]])
        # PhysX state is deliberately reset by learn(); this is an optimizer
        # resume, not an exact replay of the simulator trajectory.
        return payload
