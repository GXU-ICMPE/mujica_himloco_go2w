"""CPU integration tests for runner, auto-reset boundaries, resume and export."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from mujica.runner import MUJICARunner
from mujica.smoke import MockVecEnv, run_smoke, smoke_config
from mujica.skills import SKILL_NAMES, SKILL_VALUES, TASK_CONTRACT_VERSION


class FirstCoordinateValue(nn.Module):
    def forward(self, critic_obs):
        return critic_obs[:, :1]


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_s1_resume_s2_freeze_and_torchscript_export(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = run_smoke(directory)
            self.assertEqual(summary["status"], "PASS")
            self.assertEqual(summary["s1_iterations"], 2)
            self.assertEqual(summary["resumed_next_iteration"], 3)
            self.assertEqual(summary["s2_iterations"], 1)
            self.assertTrue(summary["s2_low_level_frozen"])
            for stage in ("s1", "s2"):
                self.assertLessEqual(summary["export_max_abs_error"][stage], 1e-6)
                self.assertTrue((Path(directory) / ("smoke_%s.pt" % stage)).is_file())
                metadata = json.loads((Path(directory) / ("smoke_%s.json" % stage)).read_text())
                self.assertTrue(metadata["smoke_fixture"])
                self.assertEqual(metadata["has_selector"], stage == "s2")
                self.assertEqual(metadata["skill_names"], list(SKILL_NAMES))
                self.assertEqual(metadata["skill_values"], SKILL_VALUES)
                self.assertEqual(metadata["task_contract_version"], TASK_CONTRACT_VERSION)

    def test_true_terminal_bootstrap_current_targets_and_successor_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            env = MockVecEnv()
            config = smoke_config()
            runner = MUJICARunner(env, config, directory)
            runner.model.critic = FirstCoordinateValue()
            captured = {}

            def capture_update(storage):
                captured.update({key: value.clone() for key, value in storage.data.items()})
                return {"capture_only": 1.0}

            runner.algorithm.update = capture_update
            with contextlib.redirect_stdout(io.StringIO()):
                runner.learn(1)
            if runner.writer:
                runner.writer.close()
            self.assertEqual(len(env.step_records), config["steps_per_env"])
            seen_timeout, seen_terminal = False, False
            for step, record in enumerate(env.step_records):
                expected_reward = (record["rewards"] + config["ppo"]["gamma"] *
                                   record["successor_critic"][:, 0] * record["timeouts"])
                self.assertTrue(torch.allclose(captured["rewards"][step], expected_reward))
                self.assertTrue(torch.equal(captured["successor_obs"][step],
                                             record["successor_critic"][:, :58]))
                self.assertTrue(torch.equal(captured["velocity_targets"][step],
                                             record["critic_before"][:, 58:61]))
                self.assertTrue(torch.equal(captured["collision_targets"][step],
                                             record["critic_before"][:, 61:79]))
                self.assertTrue(torch.equal(captured["wheel_targets"][step],
                                             record["critic_before"][:, 79:83]))
                if record["timeouts"].any():
                    seen_timeout = True
                    ids = record["timeouts"]
                    self.assertTrue((record["successor_critic"][ids, 0] > 0).all())
                    self.assertTrue((record["reset_critic"][ids, 0] < 0).all())
                if (record["dones"] & ~record["timeouts"]).any():
                    seen_terminal = True
                if step + 1 < len(env.step_records):
                    self.assertTrue(torch.equal(captured["hidden"][step + 1][record["dones"]],
                                                 torch.zeros_like(captured["hidden"][step + 1][record["dones"]])))
            self.assertTrue(seen_timeout)
            self.assertTrue(seen_terminal)

    def test_selector_strips_skill_from_all_history_frames(self):
        history = torch.arange(8 * 6 * 58).reshape(8, 6, 58).float()
        expected = history[:, :, :57].flatten(1)
        actual = MUJICARunner.selector_history(history.flatten(1))
        self.assertTrue(torch.equal(expected, actual))
        history[:, :, 57] = 99999
        self.assertTrue(torch.equal(expected, MUJICARunner.selector_history(history)))


if __name__ == "__main__":
    unittest.main()
