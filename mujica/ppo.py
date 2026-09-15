# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""HIMLoco-style PPO for Gaussian S1 or categorical S2, without P3O.

Clipped surrogate, clipped value loss, entropy bonus and adaptive KL learning
rate follow the base repository's HIMPPO. Rollout estimator outputs are held
fixed inside each PPO update; a separate optimizer fits the estimator. No
estimator parameter belongs to the policy optimizer, and no PPO loss flows
through the estimator. This makes likelihood ratios well-defined for the
features under which each rollout action was actually sampled.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn

from .storage import RolloutStorage


class PPO:
    def __init__(self, actor_critic: nn.Module, num_learning_epochs: int = 5,
                 num_mini_batches: int = 4, clip_param: float = 0.2,
                 gamma: float = 0.998, lam: float = 0.95,
                 value_loss_coef: float = 1.0, entropy_coef: float = 0.0,
                 learning_rate: float = 1e-3, max_grad_norm: float = 1.0,
                 use_clipped_value_loss: bool = True,
                 schedule: str = "fixed", desired_kl: float = 0.01,
                 estimator_learning_rate: float = 1e-3,
                 estimator_max_grad_norm: float = 10.0, device=None):
        if num_learning_epochs < 1 or num_mini_batches < 1:
            raise ValueError("Learning epochs and minibatches must be positive")
        if learning_rate <= 0 or estimator_learning_rate <= 0:
            raise ValueError("Learning rates must be positive")
        if schedule not in ("fixed", "adaptive"):
            raise ValueError("schedule must be fixed or adaptive")
        if clip_param <= 0 or max_grad_norm <= 0 or estimator_max_grad_norm <= 0:
            raise ValueError("Clipping and gradient limits must be positive")
        self.actor_critic = actor_critic
        if device is not None:
            self.actor_critic.to(device)
        self.device = next(actor_critic.parameters()).device
        self.policy_parameters = [parameter for parameter in actor_critic.policy_parameters()
                                  if parameter.requires_grad]
        if not self.policy_parameters:
            raise ValueError("PPO received a completely frozen policy")
        self.optimizer = torch.optim.Adam(self.policy_parameters, lr=learning_rate)
        self.estimator_optimizer = None
        estimator = getattr(actor_critic, "estimator", None)
        self.estimator_parameters = []
        if estimator is not None:
            self.estimator_parameters = [parameter for parameter in estimator.parameters()
                                         if parameter.requires_grad]
            if self.estimator_parameters:
                self.estimator_optimizer = torch.optim.Adam(
                    self.estimator_parameters, lr=estimator_learning_rate)
        if {id(p) for p in self.policy_parameters} & {id(p) for p in self.estimator_parameters}:
            raise ValueError("Policy and estimator optimizers must not share parameters")
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.clip_param = clip_param
        self.gamma, self.lam = gamma, lam
        self.value_loss_coef, self.entropy_coef = value_loss_coef, entropy_coef
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.estimator_max_grad_norm = estimator_max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.schedule, self.desired_kl = schedule, desired_kl

    @torch.no_grad()
    def _kl(self, batch: Dict[str, Tensor], result: Dict[str, Tensor]) -> Tensor:
        if self.actor_critic.distribution_kind == "gaussian":
            old_mean, old_std = batch["old_mean"], batch["old_std"]
            mean, std = result["mean"], result["std"]
            kl = (torch.log(std / old_std)
                  + (old_std.square() + (old_mean - mean).square()) / (2 * std.square())
                  - 0.5).sum(-1)
        elif self.actor_critic.distribution_kind == "categorical":
            old_log_probs = torch.log_softmax(batch["old_logits"], dim=-1)
            new_log_probs = torch.log_softmax(result["logits"], dim=-1)
            kl = (old_log_probs.exp() * (old_log_probs - new_log_probs)).sum(-1)
        else:
            raise ValueError("Unsupported policy distribution kind")
        return kl.mean().clamp_min(0)

    def _adapt_learning_rate(self, kl: float) -> None:
        if self.schedule != "adaptive" or self.desired_kl is None:
            return
        if kl > self.desired_kl * 2.0:
            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
        elif 0.0 < kl < self.desired_kl / 2.0:
            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate

    def update(self, storage: RolloutStorage) -> Dict[str, float]:
        self.actor_critic.train()
        estimator_keys = {"history", "hidden", "velocity_targets", "collision_targets",
                          "wheel_targets", "successor_obs"}
        if self.estimator_optimizer is not None:
            missing = estimator_keys - set(storage.data)
            if missing:
                raise ValueError("Estimator training requires transition fields %s" % sorted(missing))
        required_distribution = ({"old_mean", "old_std"}
                                 if self.actor_critic.distribution_kind == "gaussian"
                                 else {"old_logits"})
        if not required_distribution.issubset(storage.data):
            raise ValueError("Rollout lacks old policy distribution parameters")
        totals: Dict[str, float] = {}
        total_samples, updates = 0, 0
        for batch in storage.minibatches(self.num_mini_batches, self.num_learning_epochs):
            count = batch["actions"].shape[0]
            result = self.actor_critic.evaluate_actions(
                batch["policy_inputs"].detach(), batch["critic_obs"], batch["actions"])
            kl = self._kl(batch, result)
            if not torch.isfinite(kl):
                raise FloatingPointError("Nonfinite PPO KL divergence")
            self._adapt_learning_rate(kl.item())

            advantages = batch["advantages"].reshape(-1)
            ratio = torch.exp(result["log_probs"] - batch["log_probs"].reshape(-1))
            surrogate = -advantages * ratio
            surrogate_clipped = -advantages * ratio.clamp(1.0 - self.clip_param,
                                                           1.0 + self.clip_param)
            surrogate_loss = torch.maximum(surrogate, surrogate_clipped).mean()
            values = result["values"].reshape(-1)
            old_values = batch["values"].reshape(-1)
            returns = batch["returns"].reshape(-1)
            value_losses = (values - returns).square()
            if self.use_clipped_value_loss:
                value_clipped = old_values + (values - old_values).clamp(
                    -self.clip_param, self.clip_param)
                value_losses = torch.maximum(value_losses, (value_clipped - returns).square())
            value_loss = value_losses.mean()
            entropy = result["entropy"].mean()
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite PPO loss; inspect observations, rewards and actions")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(self.policy_parameters,
                                                     self.max_grad_norm,
                                                     error_if_nonfinite=True)
            self.optimizer.step()
            metrics = {"policy_loss": surrogate_loss.detach(), "value_loss": value_loss.detach(),
                       "entropy": entropy.detach(), "kl": kl,
                       "policy_grad_norm": gradient_norm.detach(),
                       "clip_fraction": ((ratio.detach() - 1).abs() > self.clip_param).float().mean()}

            if self.estimator_optimizer is not None:
                estimator_loss, estimator_metrics = self.actor_critic.estimator.loss(
                    batch["history"], batch["hidden"], batch["velocity_targets"],
                    batch["collision_targets"], batch["wheel_targets"], batch["successor_obs"])
                if not torch.isfinite(estimator_loss):
                    raise FloatingPointError("Nonfinite estimator loss")
                self.estimator_optimizer.zero_grad(set_to_none=True)
                estimator_loss.backward()
                nn.utils.clip_grad_norm_(self.estimator_parameters,
                                         self.estimator_max_grad_norm,
                                         error_if_nonfinite=True)
                self.estimator_optimizer.step()
                metrics.update(estimator_metrics)
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + float(value) * count
            total_samples += count
            updates += 1
        if not total_samples:
            raise RuntimeError("No PPO minibatches were generated")
        summary = {name: value / total_samples for name, value in totals.items()}
        summary.update(learning_rate=self.learning_rate, updates=float(updates),
                       samples=float(total_samples))
        storage.clear()
        return summary

    def state_dict(self) -> dict:
        return {"optimizer": self.optimizer.state_dict(),
                "estimator_optimizer": (self.estimator_optimizer.state_dict()
                                         if self.estimator_optimizer is not None else None),
                "learning_rate": self.learning_rate}

    def load_state_dict(self, state: dict) -> None:
        self.optimizer.load_state_dict(state["optimizer"])
        self.learning_rate = state.get("learning_rate", self.optimizer.param_groups[0]["lr"])
        if self.estimator_optimizer is not None and state.get("estimator_optimizer") is not None:
            self.estimator_optimizer.load_state_dict(state["estimator_optimizer"])
