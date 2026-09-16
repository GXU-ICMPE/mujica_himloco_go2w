# X5loco reference

Adapted from the local `/home/xgy/重要文档/X5loco` working tree on 2026-09-16.
Git HEAD: `7fd83f78a84cc51dd2a164d4ec712742ae676d31`. The source checkout has
other local modifications; this records a working-tree reference, not a claim
that every file is identical to that commit.

Copyright 2026 Robot-Nav. The repository license is reproduced in
[X5loco-LICENSE](X5loco-LICENSE) (Apache-2.0).

Reference files under `source/robot_lab/robot_lab/`:

- `assets/x5.py`: asset, joint order, targets and PD limits.
- `tasks/x5_v8/parameters.py`, `state.py`, `rewards.py`, `env_cfg.py`, `commands.py`.
- `tasks/x5_v7/state.py`, `parameters.py`, `stop_pi.py`, `env_cfg.py`.
- `tasks/x5_v6/env_cfg.py` and inherited X5/base reward configurations.

Derived files: `mujica/isaaclab/x5_parameters.py`, `x5_config.py`, `x5_state.py`,
`x5_task.py`; X5 spawn adaptation in `env_cfg.py`. Modified to use MUJICA's tensor
state, heightfield samples, explicit effort controller, learner clock and
S1/S2 lifecycle instead of X5loco's manager environment.

`resources/robots/x5/` copies the selected URDF and meshes from
`X5loco/resources/x5/`. `source_manifest.json` preserves that project's asset
provenance; it is a source record, not a new verification report. No runtime
dependency on X5loco or a machine-specific source path is needed.
