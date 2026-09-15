"""CPU integration smoke test. This does not simulate or validate locomotion.

Run ``python -m mujica.smoke`` for temporary artifacts, or add ``--output DIR``
to retain tiny randomly initialized smoke checkpoints and TorchScript exports.
Those checkpoints are test fixtures, never trained robot controllers.
"""

import argparse
import contextlib
import copy
import io
import json
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

from .config import training_config
from .export import DeploymentPolicy, export_checkpoint
from .runner import MUJICARunner, load_checkpoint
from .skills import skill_metadata


class MockVecEnv:
    """Deterministic tensor fixture implementing the base seven-return API.

    Selected environments terminate, others time out. Terminal frames contain
    positive markers and auto-reset frames contain negative markers, making
    reset-boundary mistakes observable. No rigid-body dynamics run here.
    """

    def __init__(self, num_envs=8):
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.num_obs, self.num_privileged_obs, self.num_actions = 348, 270, 16
        self.history = torch.zeros(num_envs, 6, 58)
        self.privileged = torch.zeros(num_envs, 270)
        self.episode_length = torch.zeros(num_envs, dtype=torch.long)
        self.episode_limits = 2 + torch.arange(num_envs) % 3
        self.skill = torch.arange(num_envs) % 3
        self.step_records = []
        self.reset()

    def _frame(self, ids, terminal=False):
        ids = ids.long()
        frame = torch.zeros(len(ids), 58)
        frame[:, 0] = (7.0 + ids.float()) if terminal else (-1.0 - ids.float())
        frame[:, 5] = -1.0  # projected gravity
        frame[:, 6] = 0.2
        frame[:, 57] = self.skill[ids].float()
        return frame

    def _privileged_from_frame(self, frame):
        privileged = torch.zeros(self.num_envs, 270)
        privileged[:, :58] = frame
        privileged[:, 58:61] = frame[:, :3] * 0.05
        privileged[:, 61:79] = (torch.arange(18)[None, :] % 3 ==
                                torch.arange(self.num_envs)[:, None] % 3).float()
        privileged[:, 79:83] = 0.1 + frame[:, 0:1].abs() * 0.01
        privileged[:, 83:] = 0.25
        return privileged

    def reset(self):
        ids = torch.arange(self.num_envs)
        self.episode_length.zero_()
        self.skill.copy_(ids % 3)
        self.history.copy_(self._frame(ids).unsqueeze(1).expand(-1, 6, -1))
        self.privileged = self._privileged_from_frame(self.history[:, 0])
        self.step_records = []
        return self.get_observations(), self.get_privileged_observations()

    def get_observations(self):
        return self.history.flatten(1)

    def get_privileged_observations(self):
        return self.privileged

    def set_skill(self, skills):
        self.skill.copy_(skills.long().reshape(-1))
        # The command at t changes; past executed skill indicators do not.
        self.history[:, 0, -1] = self.skill.float()
        self.privileged[:, 57] = self.skill.float()

    def step(self, actions):
        if tuple(actions.shape) != (self.num_envs, 16):
            raise ValueError("Mock environment expects 16 motor actions per environment")
        if not torch.isfinite(actions).all():
            raise FloatingPointError("Nonfinite mock motor actions")
        critic_before = self.privileged.clone()
        self.episode_length += 1
        dones = self.episode_length >= self.episode_limits
        timeouts = dones & (torch.arange(self.num_envs) % 2 == 0)
        frame = self.history[:, 0].clone()
        frame[:, :3] += actions[:, :3].tanh() * 0.01
        frame[:, 9:25] = actions.tanh() * 0.05
        frame[:, 25:41] = actions.tanh() * 0.1
        frame[:, 41:57] = actions
        frame[:, 57] = self.skill.float()
        terminal_ids = dones.nonzero(as_tuple=False).flatten()
        frame[terminal_ids, 0] = 7.0 + terminal_ids.float()
        successor = self._privileged_from_frame(frame)
        terminal_critic = successor[terminal_ids].clone()
        self.history[:, 1:] = self.history[:, :-1].clone()
        self.history[:, 0] = frame
        self.history[terminal_ids] = self._frame(terminal_ids).unsqueeze(1)
        self.episode_length[terminal_ids] = 0
        self.privileged = self._privileged_from_frame(self.history[:, 0])
        rewards = 0.5 - 0.01 * actions.square().mean(dim=-1)
        self.step_records.append(dict(critic_before=critic_before,
                                     successor_critic=successor.clone(),
                                     reset_critic=self.privileged.clone(),
                                     rewards=rewards.clone(), dones=dones.clone(),
                                     timeouts=timeouts.clone(), terminal_ids=terminal_ids.clone()))
        info = {"time_outs": timeouts,
                "next_observations": successor[:, :58].clone()}
        return (self.get_observations(), self.get_privileged_observations(), rewards,
                dones, info, terminal_ids, terminal_critic)

    def export_metadata(self):
        return dict(**skill_metadata(), smoke_fixture=True, validated_locomotion=False,
                    joint_names=[leg + "_" + joint + "_joint"
                                 for leg in ("FL", "FR", "RL", "RR")
                                 for joint in ("hip", "thigh", "calf", "foot")],
                    wheel_indices=[3, 7, 11, 15], default_dof_pos=[0.0, 0.8, -1.5, 0.0] * 4,
                    p_gains=[40.0, 40.0, 40.0, 0.0] * 4,
                    d_gains=[1.0, 1.0, 1.0, 0.5] * 4,
                    torque_limits=[23.7, 23.7, 35.55, 10.0] * 4,
                    velocity_limits=[30.0, 30.0, 20.0, 50.0] * 4,
                    sim_dt=0.005, control_dt=0.02, clip_actions=100.0,
                    action_scale=0.25, vel_scale=10.0,
                    obs_scales=dict(lin_vel=2.0, ang_vel=0.25, dof_pos=1.0, dof_vel=0.05),
                    motor=dict(enabled=False))


def smoke_config(stage="s1"):
    config = training_config(stage)
    config["model"].update(actor_hidden_dims=[32, 16], critic_hidden_dims=[32, 16],
                           estimator_kwargs=dict(gru_dim=16, encoder_hidden_dims=[32],
                                                 reference_hidden_dims=[32], num_prototypes=8))
    config["selector"].update(actor_hidden_dims=[32, 16], critic_hidden_dims=[32, 16])
    config["ppo"].update(num_learning_epochs=1, num_mini_batches=3,
                         learning_rate=3e-4, schedule="fixed")
    config.update(steps_per_env=4, iterations=2 if stage == "s1" else 1, save_interval=1)
    return config


def assert_export_parity(runner, checkpoint, output):
    scripted, metadata = export_checkpoint(checkpoint, output)
    reloaded = torch.jit.load(str(output)).eval()
    selector = runner.model if runner.stage == "s2" else None
    eager = DeploymentPolicy(runner.low_model, selector).eval()
    history = torch.randn(8, 6, 58)
    history[:, :, -1] = torch.randint(0, 3, (8, 6)).float()
    hidden = runner.low_model.initial_hidden(8)
    max_error = 0.0
    with torch.no_grad():
        for step in range(5):
            override = ((torch.arange(8) + step) % 3).long()
            if selector is not None:
                override[::2] = -1
            expected = eager(history.flatten(1), hidden, override)
            actual = scripted(history.flatten(1), hidden, override)
            loaded = reloaded(history.flatten(1), hidden, override)
            if selector is not None:
                selected = selector.act_inference(history[:, :, :57])
                selected = torch.where(override >= 0, override, selected)
            else:
                selected = override
            low_history = history.clone()
            low_history[:, 0, -1] = selected.float()
            low_actions, low_hidden = runner.low_model.act_inference(low_history, hidden)
            reference = (low_actions, low_hidden, selected)
            for left, right, persisted, original in zip(expected, actual, loaded, reference):
                if not (torch.allclose(left, right, atol=1e-6, rtol=1e-6)
                        and torch.allclose(left, persisted, atol=1e-6, rtol=1e-6)
                        and torch.allclose(left, original, atol=1e-6, rtol=1e-6)):
                    raise AssertionError("Eager, saved TorchScript, or low-level inference diverged")
                max_error = max(max_error, float((left.float() - right.float()).abs().max()))
            hidden = actual[1].clone()
            hidden[step % 8] = 0  # explicit per-env recurrent reset at deployment
            history[:, 1:] = history[:, :-1].clone()
            history[:, 0, 41:57] = actual[0]
            history[:, 0, -1] = actual[2].float()
    if not metadata.get("smoke_fixture"):
        raise AssertionError("Smoke exports must remain visibly identified as test fixtures")
    return max_error


def _close_writer(runner):
    if runner.writer:
        runner.writer.close()


def _run_in_directory(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(11)
    np.random.seed(11)
    random.seed(11)
    config = smoke_config("s1")
    s1 = MUJICARunner(MockVecEnv(), config, output_dir / "s1")
    s1.learn(2)
    s1_checkpoint = output_dir / "s1" / "last.pt"
    saved = load_checkpoint(s1_checkpoint)
    if saved["iteration"] != 2 or saved["total_steps"] != 64:
        raise AssertionError("S1 checkpoint iteration/step counters are incorrect")
    resumed = MUJICARunner(MockVecEnv(), copy.deepcopy(config), output_dir / "resumed")
    resumed.load(s1_checkpoint)
    if resumed.iteration != 2 or resumed.total_steps != 64:
        raise AssertionError("Resume failed to restore counters")
    resumed.learn(1)
    if resumed.iteration != 3 or resumed.total_steps != 96:
        raise AssertionError("Resume repeated or skipped an iteration")
    resume_metrics = [json.loads(line) for line in
                      (output_dir / "resumed" / "metrics.jsonl").read_text().splitlines()]
    if resume_metrics[-1]["iteration"] != 3:
        raise AssertionError("Resumed logging has the wrong next iteration")
    s2 = MUJICARunner(MockVecEnv(), smoke_config("s2"), output_dir / "s2",
                     low_level=s1_checkpoint)
    frozen_before = {name: value.clone() for name, value in s2.low_model.state_dict().items()}
    s2.learn(1)
    frozen = all(torch.equal(value, s2.low_model.state_dict()[name])
                 for name, value in frozen_before.items())
    if not frozen or any(parameter.requires_grad for parameter in s2.low_model.parameters()):
        raise AssertionError("S2 changed low-level actor/critic/estimator state")
    s1_error = assert_export_parity(s1, s1_checkpoint, output_dir / "smoke_s1.pt")
    s2_error = assert_export_parity(s2, output_dir / "s2" / "last.pt", output_dir / "smoke_s2.pt")
    for runner in (s1, resumed, s2):
        _close_writer(runner)
    summary = dict(status="PASS", scope="CPU tensor pipeline only; no locomotion validation",
                   num_envs=8, s1_iterations=s1.iteration, resumed_next_iteration=resumed.iteration,
                   s2_iterations=s2.iteration, s2_low_level_frozen=frozen,
                   export_max_abs_error={"s1": s1_error, "s2": s2_error},
                   checkpoints="SMOKE FIXTURES: unsuitable for robot deployment")
    (output_dir / "smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_smoke(output_dir=None):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        # Runner metric lines remain in metrics.jsonl; CLI prints one final JSON.
        with contextlib.redirect_stdout(io.StringIO()):
            if output_dir is None:
                with tempfile.TemporaryDirectory(prefix="mujica_smoke_") as directory:
                    summary = _run_in_directory(directory)
                summary["artifacts_retained"] = False
            else:
                summary = _run_in_directory(output_dir)
                summary["artifacts_retained"] = True
                summary["output_dir"] = str(Path(output_dir).resolve())
        return summary
    finally:
        torch.set_num_threads(old_threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="Keep smoke fixtures here; omitted uses temporary files")
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.output), indent=2))


if __name__ == "__main__":
    main()
