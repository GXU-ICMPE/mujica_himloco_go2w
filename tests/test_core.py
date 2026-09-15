"""CPU-only invariants for the PPO / recurrent estimator / selector boundary."""

import math
import unittest

import torch

from mujica.models import MultiSkillActorCritic, SelectorActorCritic, StateEstimator, sinkhorn
from mujica.ppo import PPO
from mujica.storage import RolloutStorage


def small_model():
    return MultiSkillActorCritic(actor_hidden_dims=(24,), critic_hidden_dims=(24,),
                                estimator_kwargs=dict(gru_dim=12,
                                    encoder_hidden_dims=(24,),
                                    reference_hidden_dims=(24,), num_prototypes=7))


def collect(model, steps=3, envs=3):
    storage = RolloutStorage(envs, steps)
    is_gaussian = model.distribution_kind == "gaussian"
    hidden = model.initial_hidden(envs) if is_gaussian else None
    for t in range(steps):
        history = torch.randn(envs, 6, 58 if is_gaussian else 57)
        critic = torch.randn(envs, 270)
        output = model.act(history, critic, hidden)
        fields = {name: output[name] for name in
                  ("actions", "values", "log_probs", "policy_inputs")}
        fields.update(critic_obs=critic, rewards=torch.randn(envs),
                      dones=torch.tensor([t == 1] + [False] * (envs - 1)))
        if is_gaussian:
            fields.update(old_mean=output["old_mean"], old_std=output["old_std"],
                          history=history, hidden=hidden,
                          velocity_targets=torch.randn(envs, 3),
                          collision_targets=torch.randint(0, 2, (envs, 18)).float(),
                          wheel_targets=torch.rand(envs, 4),
                          successor_obs=torch.randn(envs, 58))
            hidden = model.reset_hidden(output["hidden_next"], fields["dones"])
        else:
            fields["old_logits"] = output["old_logits"]
        storage.add(**fields)
    storage.compute_returns(torch.zeros(envs), gamma=0.99, lam=0.95)
    return storage


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(7)

    def test_recurrent_history_and_partial_reset(self):
        model = small_model()
        history = torch.randn(3, 6, 58)
        history[:, 0, -1] = torch.tensor([0, 1, 2])
        hidden = model.initial_hidden(3)
        first = model.act(history, torch.zeros(3, 270), hidden, deterministic=True)
        self.assertEqual(tuple(first["actions"].shape), (3, 16))
        self.assertEqual(tuple(first["features"].shape), (3, 41))
        self.assertTrue(torch.equal(first["policy_inputs"][:, :58], history[:, 0]))
        repeated = model.act(history.flatten(1), torch.zeros(3, 270), hidden, deterministic=True)
        self.assertTrue(torch.allclose(first["actions"], repeated["actions"]))
        next_output = model.act(history, torch.zeros(3, 270), first["hidden_next"], deterministic=True)
        self.assertFalse(torch.allclose(first["hidden_next"], next_output["hidden_next"]))
        reset = model.reset_hidden(first["hidden_next"], torch.tensor([True, False, True]))
        self.assertEqual(float(reset[[0, 2]].abs().sum()), 0.0)
        self.assertTrue(torch.equal(reset[1], first["hidden_next"][1]))

    def test_positive_standard_deviation_and_fixed_rollout_features(self):
        model = small_model()
        with torch.no_grad():
            model.log_std.fill_(-100)
        output = model.act(torch.randn(3, 6, 58), torch.randn(3, 270), model.initial_hidden(3))
        self.assertTrue((output["old_std"] > 0).all())
        critic = torch.randn(3, 270)
        before = model.evaluate_actions(output["policy_inputs"], critic, output["actions"])
        with torch.no_grad():
            for parameter in model.estimator.parameters():
                parameter.add_(torch.randn_like(parameter) * 3)
        after = model.evaluate_actions(output["policy_inputs"], critic, output["actions"])
        self.assertTrue(torch.equal(before["log_probs"], after["log_probs"]))
        self.assertTrue(torch.equal(output["log_probs"], after["log_probs"]))

    def test_estimator_losses_reach_all_heads_and_reference(self):
        estimator = small_model().estimator
        hidden = estimator.initial_hidden(4).requires_grad_()
        history = torch.randn(4, 6, 58, requires_grad=True)
        successor = torch.randn(4, 58, requires_grad=True)
        loss, metrics = estimator.loss(history, hidden, torch.randn(4, 3),
                                       torch.randint(0, 2, (4, 18)).float(),
                                       torch.rand(4, 4), successor)
        loss.backward()
        self.assertEqual(set(metrics), {"estimator_loss", "velocity_loss", "collision_loss",
                                        "wheel_loss", "swav_loss"})
        self.assertIsNone(hidden.grad)  # deliberate one-step truncated BPTT
        self.assertIsNone(history.grad)
        self.assertIsNone(successor.grad)
        for module in (estimator.encoder, estimator.gru, estimator.prediction,
                       estimator.reference_encoder):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                for p in module.parameters()))
        self.assertGreater(float(estimator.prototypes.grad.abs().sum()), 0)

    def test_sinkhorn_extreme_scores_are_finite(self):
        for scores in (torch.randn(11, 7), torch.tensor([[1e6, -1e6, 0.0]]),
                       torch.zeros(1, 32)):
            assignments = sinkhorn(scores)
            self.assertTrue(torch.isfinite(assignments).all())
            self.assertTrue(torch.allclose(assignments.sum(-1), torch.ones(scores.shape[0]),
                                           atol=1e-5))
        with self.assertRaises(FloatingPointError):
            sinkhorn(torch.tensor([[float("nan")]]))

    def test_all_rollout_samples_included_and_add_copies(self):
        storage = collect(small_model(), steps=3, envs=3)
        storage.data["sample_id"] = torch.arange(9).reshape(3, 3)
        batches = list(storage.minibatches(num_mini_batches=4, num_learning_epochs=2))
        self.assertEqual([len(batch["actions"]) for batch in batches], [3, 2, 2, 2] * 2)
        for epoch in range(2):
            ids = torch.cat([batch["sample_id"] for batch in batches[epoch*4:(epoch+1)*4]])
            self.assertTrue(torch.equal(ids.sort().values, torch.arange(9)))
        self.assertFalse(storage.data["policy_inputs"].requires_grad)

    def test_gae_stops_at_reset_and_bootstraps_timeout(self):
        storage = RolloutStorage(2, 2)
        # Env 0 truly terminates; env 1 times out with V(terminal)=10.
        # Timeout reward is augmented by caller BEFORE it enters storage.
        for rewards, values, dones in (
            (torch.tensor([1., 1. + .9 * 10.]), torch.tensor([2., 2.]), torch.tensor([True, True])),
            (torch.tensor([100., 100.]), torch.tensor([3., 3.]), torch.tensor([False, False]))):
            storage.add(policy_inputs=torch.zeros(2, 1), critic_obs=torch.zeros(2, 1),
                        actions=torch.zeros(2, 1), values=values, log_probs=torch.zeros(2),
                        rewards=rewards, dones=dones)
        storage.compute_returns(torch.tensor([4., 4.]), gamma=.9, lam=1.)
        self.assertTrue(torch.allclose(storage.data["returns"][0], torch.tensor([1., 10.])))
        self.assertTrue(torch.allclose(storage.data["returns"][1], torch.tensor([103.6, 103.6])))

    def test_s1_ppo_updates_actor_and_estimator_separately(self):
        model = small_model()
        storage = collect(model)
        optimizer = PPO(model, num_learning_epochs=2, num_mini_batches=4,
                        schedule="adaptive", desired_kl=.01)
        actor_before = [p.detach().clone() for p in model.actor.parameters()]
        estimator_before = [p.detach().clone() for p in model.estimator.parameters()]
        metrics = optimizer.update(storage)
        self.assertEqual(metrics["samples"], 18)
        self.assertEqual(metrics["updates"], 8)
        self.assertEqual(storage.step, 0)
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
        self.assertTrue(any(not torch.equal(before, after)
                            for before, after in zip(actor_before, model.actor.parameters())))
        self.assertTrue(any(not torch.equal(before, after)
                            for before, after in zip(estimator_before, model.estimator.parameters())))
        policy_ids = {id(p) for group in optimizer.optimizer.param_groups for p in group["params"]}
        estimator_ids = {id(p) for group in optimizer.estimator_optimizer.param_groups for p in group["params"]}
        self.assertFalse(policy_ids & estimator_ids)

    def test_s2_selector_excludes_every_skill_and_preserves_frozen_low_level(self):
        low_level = small_model().freeze()
        before = {key: value.clone() for key, value in low_level.state_dict().items()}
        selector = SelectorActorCritic(actor_hidden_dims=(24,), critic_hidden_dims=(24,))
        with self.assertRaises(ValueError):
            selector.act(torch.randn(3, 6, 58), torch.randn(3, 270))
        storage = collect(selector)
        self.assertEqual(storage.data["actions"].dtype, torch.int64)
        self.assertTrue(((storage.data["actions"] >= 0) & (storage.data["actions"] < 3)).all())
        optimizer = PPO(selector, num_learning_epochs=2, num_mini_batches=4,
                        schedule="adaptive")
        metrics = optimizer.update(storage)
        self.assertIsNone(optimizer.estimator_optimizer)
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
        for key, value in low_level.state_dict().items():
            self.assertTrue(torch.equal(before[key], value))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in low_level.parameters()))


if __name__ == "__main__":
    unittest.main()
