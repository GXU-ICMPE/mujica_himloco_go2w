"""Export a deterministic policy with explicit recurrent state, without critic."""
import argparse
import copy
import json
from pathlib import Path
from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .models import MultiSkillActorCritic, SelectorActorCritic
from .runner import load_checkpoint
from .skills import SKILL_VALUES, validate_skill_metadata


class DeploymentPolicy(nn.Module):
    def __init__(self, low, selector=None):
        super().__init__()
        self.encoder = copy.deepcopy(low.estimator.encoder)
        self.gru = copy.deepcopy(low.estimator.gru)
        self.prediction = copy.deepcopy(low.estimator.prediction)
        self.actor = copy.deepcopy(low.actor)
        self.has_selector = selector is not None
        self.selector = copy.deepcopy(selector.actor) if selector is not None else nn.Linear(342, 3)
        self.history_len = low.history_len
        self.frame_dim = low.frame_dim
        self.collision_dim = low.estimator.collision_dim
        self.wheel_dim = low.estimator.wheel_dim
        self.latent_dim = low.estimator.latent_dim

    def forward(self, history: Tensor, hidden: Tensor,
                skill_override: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        frames = history.reshape(-1, self.history_len, self.frame_dim).clone()
        if bool(torch.any((skill_override < -1) | (skill_override > 2))):
            raise ValueError("skill_override must contain -1, 0, 1, or 2")
        if self.has_selector:
            probabilities = self.selector(frames[:, :, :-1].reshape(frames.shape[0], -1))
            selected = probabilities.argmax(dim=-1)
        else:
            if bool(torch.any(skill_override < 0)):
                raise ValueError("S1 export has no selector; provide explicit skill")
            selected = torch.zeros_like(skill_override)
        selected = torch.where(skill_override >= 0, skill_override, selected).long()
        frames[:, 0, -1] = selected.to(frames.dtype)
        hidden_next = self.gru(self.encoder(frames.flatten(1)), hidden)
        raw = self.prediction(hidden_next)
        velocity = raw[:, :3]
        collision = raw[:, 3:3 + self.collision_dim].sigmoid()
        wheel = raw[:, 3 + self.collision_dim:3 + self.collision_dim + self.wheel_dim]
        latent = F.normalize(raw[:, -self.latent_dim:], p=2.0, dim=-1, eps=1e-8)
        features = torch.cat((velocity, collision, wheel, latent), dim=-1)
        action = self.actor(torch.cat((frames[:, 0], features), dim=-1))
        return action, hidden_next, selected


def export_checkpoint(checkpoint, output):
    saved = load_checkpoint(checkpoint)
    validate_skill_metadata(saved.get("metadata", {}))
    low = MultiSkillActorCritic(**saved["config"]["model"])
    low.load_state_dict(saved["low_model"])
    selector = None
    if saved["config"]["stage"] == "s2":
        selector = SelectorActorCritic(**saved["config"]["selector"])
        selector.load_state_dict(saved["selector"])
    module = DeploymentPolicy(low, selector).cpu().eval()
    scripted = torch.jit.script(module)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(output))
    metadata = copy.deepcopy(saved.get("metadata", {}))
    metadata.update(format_version=1, stage=saved["config"]["stage"],
                    has_selector=selector is not None, frame_dim=low.frame_dim,
                    history_len=low.history_len, hidden_dim=low.estimator.gru_dim,
                    history_order="newest_first", skill_values=dict(SKILL_VALUES),
                    control_decimation=4, clip_observations=100.0)
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return scripted, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    export_checkpoint(args.checkpoint, args.output)
    print("Exported " + args.output + " and matching .json metadata")


if __name__ == "__main__":
    main()
