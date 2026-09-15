"""Simulator-independent MUJICA networks.

The paper specifies the signal paths, H=6 and a GRU, but does not publish
network widths or the latent width. Defaults below are explicit engineering
choices. Equation (5) is implemented with a *persistent* GRUCell state. The
caller owns that state and must reset only rows whose episodes have ended.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.distributions import Categorical, Normal


def mlp(input_dim: int, output_dim: int, hidden_dims: Sequence[int],
        activation: str = "elu") -> nn.Sequential:
    activations = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh,
                   "silu": nn.SiLU, "lrelu": nn.LeakyReLU}
    if activation not in activations:
        raise ValueError("Unsupported activation: %s" % activation)
    layers = []
    for width in hidden_dims:
        layers.extend([nn.Linear(input_dim, width), activations[activation]()])
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


def reshape_history(history: Tensor, history_len: int, frame_dim: int) -> Tensor:
    """Validate newest-first history; never silently discard a skill channel."""
    if history.ndim == 2 and history.shape[1] == history_len * frame_dim:
        return history.reshape(history.shape[0], history_len, frame_dim)
    if history.ndim == 3 and tuple(history.shape[1:]) == (history_len, frame_dim):
        return history
    raise ValueError("Expected history [B,%d,%d] or [B,%d], got %s" %
                     (history_len, frame_dim, history_len * frame_dim,
                      tuple(history.shape)))


@torch.no_grad()
def sinkhorn(scores: Tensor, epsilon: float = 0.05,
             iterations: int = 3) -> Tensor:
    """Balanced SwAV assignments in log space, avoiding exp overflow.

    Returns [batch, prototypes], with each sample summing to one. Even when
    batch < prototype count, no sample is dropped and denominators stay finite.
    """
    if scores.ndim != 2 or min(scores.shape) == 0:
        raise ValueError("Sinkhorn needs a nonempty [batch, prototypes] tensor")
    if epsilon <= 0 or iterations < 1:
        raise ValueError("epsilon and iterations must be positive")
    if not torch.isfinite(scores).all():
        raise FloatingPointError("Nonfinite SwAV scores")
    batch, prototypes = scores.shape
    # float64 protects intermediate subtraction for unusually large scores.
    log_q = scores.detach().double().T / epsilon
    log_q = log_q - torch.logsumexp(log_q.reshape(-1), dim=0)
    for _ in range(iterations):
        log_q = log_q - torch.logsumexp(log_q, dim=1, keepdim=True)
        log_q = log_q - math.log(prototypes)
        log_q = log_q - torch.logsumexp(log_q, dim=0, keepdim=True)
        log_q = log_q - math.log(batch)
    return (log_q + math.log(batch)).exp().T.to(scores.dtype)


class StateEstimator(nn.Module):
    """History MLP -> persistent GRU -> v, collision, wheel distance, latent.

    Supervised predictions are aligned to t. Only the contrastive reference
    encoder sees the actual t+1 observation, including terminal observations.
    Estimator training replays detached rollout hidden states (one-step
    truncated BPTT); recurrent rollout state still persists across all steps.
    """

    def __init__(self, frame_dim: int = 58, history_len: int = 6,
                 collision_dim: int = 18, wheel_dim: int = 4,
                 latent_dim: int = 16, gru_dim: int = 64,
                 encoder_hidden_dims: Sequence[int] = (128, 64),
                 reference_hidden_dims: Sequence[int] = (128, 64),
                 num_prototypes: int = 32, temperature: float = 3.0,
                 sinkhorn_epsilon: float = 0.05, sinkhorn_iterations: int = 3,
                 velocity_loss_coef: float = 1.0,
                 collision_loss_coef: float = 1.0,
                 wheel_loss_coef: float = 1.0,
                 swav_loss_coef: float = 1.0,
                 activation: str = "elu"):
        super().__init__()
        if min(frame_dim, history_len, collision_dim, wheel_dim, latent_dim,
               gru_dim, num_prototypes) <= 0 or temperature <= 0:
            raise ValueError("All estimator dimensions and temperature must be positive")
        self.frame_dim, self.history_len = frame_dim, history_len
        self.collision_dim, self.wheel_dim = collision_dim, wheel_dim
        self.latent_dim, self.gru_dim = latent_dim, gru_dim
        self.feature_dim = 3 + collision_dim + wheel_dim + latent_dim
        self.encoder = mlp(history_len * frame_dim, gru_dim,
                           encoder_hidden_dims, activation)
        self.gru = nn.GRUCell(gru_dim, gru_dim)
        self.prediction = nn.Linear(gru_dim, self.feature_dim)
        self.reference_encoder = mlp(frame_dim, latent_dim,
                                     reference_hidden_dims, activation)
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, latent_dim))
        self.temperature = temperature
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_iterations = sinkhorn_iterations
        self.loss_coefficients = (velocity_loss_coef, collision_loss_coef,
                                  wheel_loss_coef, swav_loss_coef)

    def initial_hidden(self, num_envs: int, device=None) -> Tensor:
        parameter = next(self.parameters())
        return torch.zeros(num_envs, self.gru_dim,
                           device=device if device is not None else parameter.device,
                           dtype=parameter.dtype)

    @staticmethod
    def reset_hidden(hidden: Tensor, dones: Tensor) -> Tensor:
        return hidden * (~dones.reshape(-1).bool()).to(hidden.dtype).unsqueeze(-1)

    def forward(self, history: Tensor, hidden: Tensor) -> Dict[str, Tensor]:
        frames = reshape_history(history, self.history_len, self.frame_dim)
        if tuple(hidden.shape) != (frames.shape[0], self.gru_dim):
            raise ValueError("GRU hidden shape does not match batch/gru_dim")
        hidden_next = self.gru(self.encoder(frames.flatten(1)), hidden)
        raw = self.prediction(hidden_next)
        velocity, collision_logits, wheel_distances, latent = torch.split(
            raw, (3, self.collision_dim, self.wheel_dim, self.latent_dim), dim=-1)
        collision = collision_logits.sigmoid()
        latent = F.normalize(latent, dim=-1, eps=1e-8)
        features = torch.cat((velocity, collision, wheel_distances, latent), dim=-1)
        return {"velocity": velocity, "collision_logits": collision_logits,
                "collision": collision, "wheel_distances": wheel_distances,
                "latent": latent, "features": features, "hidden_next": hidden_next}

    def loss(self, history: Tensor, hidden: Tensor, velocity_targets: Tensor,
             collision_targets: Tensor, wheel_targets: Tensor,
             successor_obs: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        prediction = self(history.detach(), hidden.detach())
        if tuple(successor_obs.shape) != (history.shape[0], self.frame_dim):
            raise ValueError("successor_obs must be the actual t+1 [B, frame_dim] frame")
        for name, target, expected in (
            ("velocity", velocity_targets, prediction["velocity"]),
            ("collision", collision_targets, prediction["collision"]),
            ("wheel", wheel_targets, prediction["wheel_distances"]),
        ):
            if target.shape != expected.shape:
                raise ValueError("%s target shape %s != %s" %
                                 (name, tuple(target.shape), tuple(expected.shape)))
        if ((collision_targets < 0) | (collision_targets > 1)).any():
            raise ValueError("Collision targets must be probabilities/binary labels in [0, 1]")
        v_loss = F.mse_loss(prediction["velocity"], velocity_targets.detach())
        c_loss = F.binary_cross_entropy_with_logits(
            prediction["collision_logits"], collision_targets.detach())
        u_loss = F.mse_loss(prediction["wheel_distances"], wheel_targets.detach())
        reference = F.normalize(self.reference_encoder(successor_obs.detach()),
                                dim=-1, eps=1e-8)
        prototypes = F.normalize(self.prototypes, dim=-1, eps=1e-8)
        scores_online = prediction["latent"] @ prototypes.T
        scores_reference = reference @ prototypes.T
        assignments_online = sinkhorn(scores_online, self.sinkhorn_epsilon,
                                      self.sinkhorn_iterations)
        assignments_reference = sinkhorn(scores_reference, self.sinkhorn_epsilon,
                                         self.sinkhorn_iterations)
        log_online = F.log_softmax(scores_online / self.temperature, dim=-1)
        log_reference = F.log_softmax(scores_reference / self.temperature, dim=-1)
        swav = -0.5 * ((assignments_reference * log_online).sum(dim=-1).mean()
                       + (assignments_online * log_reference).sum(dim=-1).mean())
        terms = (v_loss, c_loss, u_loss, swav)
        total = sum(weight * term for weight, term in zip(self.loss_coefficients, terms))
        return total, {"estimator_loss": total.detach(), "velocity_loss": v_loss.detach(),
                       "collision_loss": c_loss.detach(), "wheel_loss": u_loss.detach(),
                       "swav_loss": swav.detach()}


class MultiSkillActorCritic(nn.Module):
    """ONE shared skill-conditioned Gaussian low-level policy (MUJICA S1).

    Scalar skill indicator is part of each observation frame. No experts,
    distillation, skill-specific actor heads or learned mixture are introduced.
    """

    distribution_kind = "gaussian"

    def __init__(self, frame_dim: int = 58, history_len: int = 6,
                 critic_dim: int = 270, action_dim: int = 16,
                 collision_dim: int = 18, latent_dim: int = 16,
                 actor_hidden_dims: Sequence[int] = (512, 256, 128),
                 critic_hidden_dims: Sequence[int] = (512, 256, 128),
                 init_noise_std: float = 1.0, activation: str = "elu",
                 estimator_kwargs: Optional[dict] = None):
        super().__init__()
        if init_noise_std <= 0:
            raise ValueError("init_noise_std must be strictly positive")
        self.frame_dim, self.history_len = frame_dim, history_len
        self.critic_dim, self.action_dim = critic_dim, action_dim
        est_kwargs = dict(frame_dim=frame_dim, history_len=history_len,
                          collision_dim=collision_dim, latent_dim=latent_dim,
                          activation=activation)
        est_kwargs.update(estimator_kwargs or {})
        for key, expected in (("frame_dim", frame_dim), ("history_len", history_len),
                              ("collision_dim", collision_dim), ("latent_dim", latent_dim)):
            if est_kwargs[key] != expected:
                raise ValueError("estimator_kwargs.%s conflicts with policy" % key)
        self.estimator = StateEstimator(**est_kwargs)
        self.policy_input_dim = frame_dim + self.estimator.feature_dim
        self.actor = mlp(self.policy_input_dim, action_dim, actor_hidden_dims, activation)
        self.critic = mlp(critic_dim, 1, critic_hidden_dims, activation)
        # Parameterize log sigma so optimizer updates cannot produce sigma <= 0.
        self.log_std = nn.Parameter(torch.full((action_dim,), math.log(init_noise_std)))

    def initial_hidden(self, num_envs: int, device=None) -> Tensor:
        return self.estimator.initial_hidden(num_envs, device)

    @staticmethod
    def reset_hidden(hidden: Tensor, dones: Tensor) -> Tensor:
        return StateEstimator.reset_hidden(hidden, dones)

    def distribution(self, policy_inputs: Tensor) -> Normal:
        mean = self.actor(policy_inputs)
        std = self.log_std.clamp(-10.0, 2.0).exp().expand_as(mean)
        return Normal(mean, std)

    def value(self, critic_obs: Tensor) -> Tensor:
        return self.critic(critic_obs).squeeze(-1)

    @torch.no_grad()
    def act(self, history: Tensor, critic_obs: Tensor, hidden: Tensor,
            deterministic: bool = False) -> Dict[str, Tensor]:
        frames = reshape_history(history, self.history_len, self.frame_dim)
        estimate = self.estimator(frames, hidden)
        policy_inputs = torch.cat((frames[:, 0], estimate["features"]), dim=-1)
        distribution = self.distribution(policy_inputs)
        actions = distribution.mean if deterministic else distribution.sample()
        return {"actions": actions, "values": self.value(critic_obs),
                "log_probs": distribution.log_prob(actions).sum(-1),
                "policy_inputs": policy_inputs, "old_mean": distribution.mean,
                "old_std": distribution.stddev,
                "hidden_next": estimate["hidden_next"], "features": estimate["features"]}

    def evaluate_actions(self, policy_inputs: Tensor, critic_obs: Tensor,
                         actions: Tensor) -> Dict[str, Tensor]:
        distribution = self.distribution(policy_inputs)
        return {"values": self.value(critic_obs),
                "log_probs": distribution.log_prob(actions).sum(-1),
                "entropy": distribution.entropy().sum(-1),
                "mean": distribution.mean, "std": distribution.stddev}

    @torch.no_grad()
    def act_inference(self, history: Tensor, hidden: Tensor) -> Tuple[Tensor, Tensor]:
        frames = reshape_history(history, self.history_len, self.frame_dim)
        estimate = self.estimator(frames, hidden)
        inputs = torch.cat((frames[:, 0], estimate["features"]), dim=-1)
        return self.actor(inputs), estimate["hidden_next"]

    def policy_parameters(self):
        yield from self.actor.parameters()
        yield from self.critic.parameters()
        yield self.log_std

    def freeze(self):
        """Freeze the whole low-level controller including the online estimator."""
        self.requires_grad_(False)
        self.eval()
        return self


class SelectorActorCritic(nn.Module):
    """MUJICA S2: proprioceptive categorical selector, with no skill input."""

    distribution_kind = "categorical"

    def __init__(self, history_len: int = 6, base_frame_dim: int = 57,
                 critic_dim: int = 270, skill_count: int = 3,
                 actor_hidden_dims: Sequence[int] = (256, 128),
                 critic_hidden_dims: Sequence[int] = (512, 256, 128),
                 activation: str = "elu"):
        super().__init__()
        self.history_len, self.base_frame_dim = history_len, base_frame_dim
        self.critic_dim, self.skill_count = critic_dim, skill_count
        self.policy_input_dim = history_len * base_frame_dim
        self.actor = mlp(self.policy_input_dim, skill_count, actor_hidden_dims, activation)
        self.critic = mlp(critic_dim, 1, critic_hidden_dims, activation)

    def distribution(self, policy_inputs: Tensor) -> Categorical:
        return Categorical(logits=self.actor(policy_inputs))

    def value(self, critic_obs: Tensor) -> Tensor:
        return self.critic(critic_obs).squeeze(-1)

    @torch.no_grad()
    def act(self, history: Tensor, critic_obs: Tensor, hidden=None,
            deterministic: bool = False) -> Dict[str, Tensor]:
        inputs = reshape_history(history, self.history_len, self.base_frame_dim).flatten(1)
        distribution = self.distribution(inputs)
        actions = distribution.probs.argmax(-1) if deterministic else distribution.sample()
        return {"actions": actions, "values": self.value(critic_obs),
                "log_probs": distribution.log_prob(actions), "policy_inputs": inputs,
                "old_logits": distribution.logits, "hidden_next": hidden}

    def evaluate_actions(self, policy_inputs: Tensor, critic_obs: Tensor,
                         actions: Tensor) -> Dict[str, Tensor]:
        distribution = self.distribution(policy_inputs)
        return {"values": self.value(critic_obs),
                "log_probs": distribution.log_prob(actions.long().reshape(-1)),
                "entropy": distribution.entropy(), "logits": distribution.logits}

    @torch.no_grad()
    def act_inference(self, history: Tensor) -> Tensor:
        inputs = reshape_history(history, self.history_len, self.base_frame_dim).flatten(1)
        return self.actor(inputs).argmax(-1)

    def policy_parameters(self):
        return self.parameters()
