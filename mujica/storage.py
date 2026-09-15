"""Detached vectorized rollout storage with complete minibatch coverage."""

from __future__ import annotations

from typing import Dict, Iterator

import torch
from torch import Tensor


class RolloutStorage:
    """One on-policy rollout; tensors are allocated on the first transition.

    Required fields are named below. Gaussian policies additionally need
    old_mean/old_std; categorical policies need old_logits. Estimator training
    additionally records history, pre-action hidden, current velocity_targets,
    collision_targets, wheel_targets and the true pre-reset successor_obs.

    Timeouts must already be bootstrapped into rewards with gamma * V(s_T).
    Both terminated and truncated episodes set dones=True, cutting GAE at
    reset boundaries. This avoids bootstrapping from reset observations.
    """

    REQUIRED = {"policy_inputs", "critic_obs", "actions", "values",
                "log_probs", "rewards", "dones"}

    def __init__(self, num_envs: int, num_steps: int, device="cpu"):
        if num_envs < 1 or num_steps < 1:
            raise ValueError("num_envs and num_steps must be positive")
        self.num_envs, self.num_steps = num_envs, num_steps
        self.device = torch.device(device)
        self.data: Dict[str, Tensor] = {}
        self.step = 0
        self._returns_ready = False

    def add(self, **fields: Tensor) -> None:
        if self.step >= self.num_steps:
            raise RuntimeError("Rollout storage is full; update/clear before adding")
        missing = self.REQUIRED - set(fields)
        if missing:
            raise ValueError("Missing transition fields: %s" % sorted(missing))
        if "advantages" in fields or "returns" in fields:
            raise ValueError("advantages and returns are computed by storage")
        if not self.data:
            for name, value in fields.items():
                if not isinstance(value, Tensor) or value.ndim < 1 or value.shape[0] != self.num_envs:
                    raise ValueError("%s must be a tensor with leading num_envs dimension" % name)
                self.data[name] = torch.empty((self.num_steps,) + tuple(value.shape),
                                              dtype=value.dtype, device=self.device)
        expected_keys = set(self.data) - {"advantages", "returns"}
        if set(fields) != expected_keys:
            raise ValueError("Transition fields must be consistent throughout a rollout")
        for name, value in fields.items():
            if tuple(value.shape) != tuple(self.data[name].shape[1:]):
                raise ValueError("Transition %s has inconsistent shape" % name)
            self.data[name][self.step].copy_(value.detach())
        self.step += 1
        self._returns_ready = False

    @torch.no_grad()
    def compute_returns(self, last_values: Tensor, gamma: float = 0.99,
                        lam: float = 0.95) -> None:
        if self.step == 0:
            raise RuntimeError("Cannot compute returns for an empty rollout")
        if not 0 <= gamma <= 1 or not 0 <= lam <= 1:
            raise ValueError("gamma and lambda must be in [0, 1]")
        if last_values.numel() != self.num_envs:
            raise ValueError("last_values needs one scalar per environment")
        values = self.data["values"][:self.step].reshape(self.step, self.num_envs)
        rewards = self.data["rewards"][:self.step].reshape(self.step, self.num_envs)
        dones = self.data["dones"][:self.step].reshape(self.step, self.num_envs).bool()
        returns = torch.empty_like(values)
        advantage = torch.zeros(self.num_envs, device=self.device, dtype=values.dtype)
        next_values = last_values.to(self.device).reshape(self.num_envs)
        for t in reversed(range(self.step)):
            continuation = (~dones[t]).to(values.dtype)
            delta = rewards[t] + gamma * next_values * continuation - values[t]
            advantage = delta + gamma * lam * continuation * advantage
            returns[t] = advantage + values[t]
            next_values = values[t]
        advantages = returns - values
        # population std also handles a single sample without NaNs.
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        self.data["returns"] = returns
        self.data["advantages"] = advantages
        self._returns_ready = True

    def minibatches(self, num_mini_batches: int = 4,
                    num_learning_epochs: int = 5) -> Iterator[Dict[str, Tensor]]:
        if not self._returns_ready:
            raise RuntimeError("Call compute_returns before requesting minibatches")
        if num_mini_batches < 1 or num_learning_epochs < 1:
            raise ValueError("Minibatch and epoch counts must be positive")
        count = self.step * self.num_envs
        batches = min(num_mini_batches, count)
        flat = {key: value[:self.step].reshape((count,) + tuple(value.shape[2:]))
                for key, value in self.data.items()}
        for _ in range(num_learning_epochs):
            permutation = torch.randperm(count, device=self.device)
            # tensor_split includes the remainder, unlike floor-sized slicing.
            for indices in torch.tensor_split(permutation, batches):
                yield {key: value[indices] for key, value in flat.items()}

    def clear(self) -> None:
        self.step = 0
        self.data.pop("returns", None)
        self.data.pop("advantages", None)
        self._returns_ready = False
