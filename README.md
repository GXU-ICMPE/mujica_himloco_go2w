<<<<<<< HEAD
# mujica_himloco_go2w
=======
# MUJICA on HIMLoco-for-Go2W · PPO版

基于完整的 [TrackinBIT/HIMLoco-for-Go2W](https://github.com/TrackinBIT/HIMLoco-for-Go2W) 修改，包含上游代码、Go2W URDF/MJCF/网格、MUJICA两阶段训练、三技能环境、配置、检查点续训、推理导出、MuJoCo验证及测试。

**算法约定：使用PPO；S1共用Go2W奖励并按地形放宽姿态约束；按MUJICA实现共享底层策略、GRU状态估计器和自动技能选择器。** 上游基座为 Isaac Gym Preview 4，当前默认后端已改为 **Isaac Lab**，保留旧后端用于对照。

当前完成 Isaac Lab 后端的源码迁移及离线检查，未附经过完整训练的策略权重。旧交付记录包含 CPU 算法/集成测试和 MuJoCo 链路验证；**本次未运行 Isaac Lab 仿真或仿真 PPO，尚未验证三技能收敛、论文成功率或真机表现**。当前任务已改为下面三类地形行走，奖励已加入复杂地形的姿态放宽，见[任务适配说明](docs/TERRAIN_TASKS_ZH.md)和[奖励现状](docs/REWARDS_ZH.md)。

| ID / 技能名 | 场景 | 预期运动方式 |
|---|---|---|
| 0 / `flat_slope` | 平地、上坡、下坡 | 轮式行走为主，无主动抬腿需求 |
| 1 / `discrete` | 随机离散障碍 | 不规则抬腿 |
| 2 / `stairs` | 上楼梯、下楼梯 | 较有规律的抬腿 |

这些运动方式是学习目标，本次未锁死腿关节或加入步态/抬腿奖励。离散障碍和楼梯共用较宽松的姿态权重、3cm 高度容忍量及停车关节容忍量，见[奖励说明](docs/REWARDS_ZH.md)。旧 Moving / Climb / Recovery 检查点的技能含义不同，程序会拒绝加载，需要重新训练 S1。

## 安装与离线检查

默认训练入口已迁到 **Isaac Lab DirectRLEnv**。当前按本机 `x3w_isaaclab` 的
Isaac Sim 5.1.0、Isaac Lab Python 包 0.54.4、Python 3.11、PyTorch 2.7.0+cu128 接口实现。
这组版本已读取确认；真实 Lab 场景尚未启动验收。

在项目根目录执行：

```bash
conda activate x3w_isaaclab
# 当前源码可直接运行；需要可编辑安装时执行：
python -m pip install -e . --no-deps
# 仅在缺少辅助依赖时安装，不替换环境已有的 Isaac Lab/Sim 和 CUDA torch
python -m pip install -r requirements-mujica.txt

# 不启动 Isaac Sim：配置、资源、关节映射和 URDF 合并检查
python -m mujica.train --check-config
python -m pytest -q tests
```

最新奖励修改按用户要求留待自行验证；当前不是完整测试通过的版本声明，具体检查历史见 [验证记录](docs/VALIDATION.md)。

后续可先做零动作物理接口检查，不执行 PPO 更新（本次未运行）：

```bash
python -m mujica.isaaclab.sim_smoke --num-envs 9 --steps 350 --terrain trimesh --headless
```

无需安装包内旧 `rsl_rl`。`mujica/` 使用自己的 PPO 和 runner；新默认依赖不包含 `isaacgym`。
现有 `legged_gym/` 与旧入口保留在 `python -m mujica.train_isaacgym`，也可显式指定
`python -m mujica.train --backend isaacgym ...`。旧环境安装说明见 [Isaac Gym 记录](docs/ISAAC_GYM_LEGACY.md)。

迁移范围、对应关系和待验证项见 [Isaac Lab 迁移说明](docs/ISAACLAB_MIGRATION_ZH.md)。

## S1：三技能联合训练

以下是后续由你运行的仿真命令，尚未在本次迁移中执行。先用小地形检查启动：

```bash
python -m mujica.train --stage s1 --num-envs 64 --terrain-rows 2 --terrain-cols 6 --headless --iterations 2 --log-dir logs/terrain_posture_v1_startup_check
```

完成仿真启动与物理接口检查后，使用默认 20×18 地形和新日志目录训练。环境数按本机显存选择；下面从 512 环境起步，不代表已做吞吐或显存验收：

```bash
python -m mujica.train --stage s1 --num-envs 512 --headless --config configs/mujica_default.json --iterations 30000 --log-dir logs/terrain_posture_v1_s1
```

显存不足可降低环境数；CLI 未指定时默认 64 个环境。默认每轮48步，PPO每轮5个epoch、4个mini-batch。`--iterations`为本次**新增**迭代数。

- 一个共享 actor 同时训练 `flat_slope`、`discrete`、`stairs`，三类任务的环境数量最多相差 1。
- 物理步 5ms，策略步 20ms；全部从正立姿态开始、立即驱动，最长 20 秒。机身碰撞结束回合，越出当前地形块提前截断。
- 默认 20 级课程、18 列地形；平地/上坡/下坡/离散障碍/上楼/下楼六种地形重复三组。S1 共用 15 项奖励，复杂地形放宽四项姿态权重；总奖励仍正截断。
- 默认电机包络是依据URDF与论文曲线形态的未校准近似，可通过`--motor-config`替换；`--disable-motor-model`改为静态力矩限幅。

## S2：训练自动技能选择器

确认S1三个技能各自具备所需能力后，再启动S2。程序能加载检查点不代表技能已经合格。

```bash
python -m mujica.train --stage s2 --num-envs 512 --headless --low-level logs/terrain_posture_v1_s1/last.pt --iterations 10000 --log-dir logs/terrain_posture_v1_s2
```

此阶段冻结整个底层策略与估计器，训练离散选择器及其特权critic。每20ms选择一次技能，使用原Go2W两项速度跟踪奖励。三个新任务统一使用正立重置、20 秒上限和机身碰撞终止规则；选择技能不改变地形和重置条件。

## 续训、查看和导出

```bash
# S1继续训练5000轮
python -m mujica.train --stage s1 --resume logs/terrain_posture_v1_s1/last.pt --num-envs 512 --headless --iterations 5000 --log-dir logs/terrain_posture_v1_s1_resume

# S2续训无需另外传S1路径
python -m mujica.train --stage s2 --resume logs/terrain_posture_v1_s2/last.pt --num-envs 512 --headless --iterations 2000 --log-dir logs/terrain_posture_v1_s2_resume

# 自动选择器播放；不加headless即可打开窗口
python -m mujica.train --stage s2 --resume logs/terrain_posture_v1_s2/last.pt --num-envs 12 --play --skill auto --play-steps 2000

# 导出选择器+底层actor+估计器及关节映射JSON
python -m mujica.export --checkpoint logs/terrain_posture_v1_s2/last.pt --output exports/mujica_go2w.pt
```

S1播放或导出后推理必须指定`flat_slope`、`discrete`、`stairs`之一，S1不含选择器。S1环境仍按实际地形任务重置，手动覆盖技能不会修改地形任务。单技能的指定姿态测试可以使用下方MuJoCo接口。

日志目录含`config.json`、`environment.json`、`metrics.jsonl`、定期检查点及`last.pt`。分别记录三类任务及六种地形的速度跟踪分数、位移、失败率、越界截断率，以及请求力矩越界统计和 S2 技能选择比例。日志字段见[任务说明](docs/TERRAIN_TASKS_ZH.md)。安装TensorBoard后也记录事件日志。续训恢复模型、优化器、迭代数、环境配置与随机数状态，但重置PhysX环境，**不等于从某一物理帧逐位重放**。训练续训不会静默更改检查点环境设置。`--resume` 只接受 Isaac Lab 检查点；旧 Gym 检查点不能作为 Lab 物理环境的续训。跨后端底层网络初始化还必须具有完全一致的关节顺序和新版技能协议元数据。

## MuJoCo Sim2Sim

MuJoCo是可选依赖，可在独立的较新Python/PyTorch环境安装。导出的`.pt`和同名`.json`应一起复制。

```bash
pip install mujoco
python -m mujica.sim2sim --policy exports/mujica_go2w.pt --metadata exports/mujica_go2w.json --skill auto --vx 0.5 --seconds 30

# 在默认平地场景固定使用平地/坡道技能（S1 导出必须手动指定技能）
python -m mujica.sim2sim --policy exports/mujica_go2w.pt --metadata exports/mujica_go2w.json --skill flat_slope --vx 0.5 --seconds 30 --headless
```

默认生成平地场景并加载包内机器人，避开上游scene中的机器专属路径；`--scene`可指定自己的坡道/离散障碍/楼梯场景；默认平地不能验证另外两类技能。映射来自JSON中的明确的策略关节名称，显式处理URDF的`foot_joint`与MJCF的`wheel_joint`差异，不假定两个仿真器索引相同。

部署接口：

```python
actions, next_hidden, skill = policy(history, hidden, skill_override)
# history: [N,6,58]或[N,348]，最新在前；初始化复制当前帧6次
# hidden: [N,hidden_dim]，见JSON；机器人reset时清零
# skill_override: [N]整数，-1自动(仅S2)，0平地及坡道，1离散障碍，2楼梯
# actions: [N,16]，JSON记录的策略关节顺序（通过名称映射到Lab关节）
```

不要直接用上游`mujoco/pdandrl.py`加载新策略；新接口额外包含GRU状态与技能选择器。

## 项目导航

| 路径 | 内容 |
|---|---|
| `mujica/models.py` | GRU估计器、SwAV、共享actor/critic、分类选择器 |
| `mujica/ppo.py`、`storage.py` | PPO、GAE、rollout及估计器监督数据 |
| `mujica/runner.py`、`train.py` | S1/S2运行、日志、检查点续训 |
| `mujica/motor.py` | 速度与小腿位置相关电机包络 |
| `mujica/isaaclab/` | 默认 Lab 环境、资产转换、地形、任务逻辑与 runner 适配 |
| `mujica/skills.py`、`terrain.py`、`locomotion_tasks.py` | 新技能协议、共享地形与任务生命周期 |
| `mujica/rewards.py` | 三类任务共用奖励、姿态放宽与回合诊断 |
| `legged_gym/envs/mujica/` | 同步新任务语义的备用 Gym 后端 |
| `mujica/export.py`、`sim2sim.py` | TorchScript导出和MuJoCo运行 |
| `mujica/smoke.py`、`tests/` | 无Isaac算法、接口、导出测试 |
| `resources/robots/go2w/` | 完整Go2W机器人资源 |
| `configs/` | 网络/PPO及未校准电机配置 |
| `docs/ARCHITECTURE_ZH.md` | 论文结构对应与工程补全边界 |
| `docs/TERRAIN_TASKS_ZH.md` | 三类新任务、地形参数、课程及日志 |
| `docs/REWARDS_ZH.md` | 当前共用奖励、实际权重和容忍参数 |
| `docs/PAPER_AUDIT_ZH.md` | 对照原文的结构审查及工程差异 |
| `docs/VALIDATION.md` | 实际验证与未验证范围 |
| `docs/README_UPSTREAM.md` | 基座原README |

原版`go2w`任务及其训练代码保留。`python -m mujica.train` 默认使用 Lab；`python -m mujica.train_isaacgym` 使用旧 Gym。

## 来源与许可

- [MUJICA论文](https://arxiv.org/abs/2605.13058)、[作者项目页](https://hyzenthlayer.github.io/mujica/)。这是独立结构复现，不是作者官方代码。
- [HIMLoco-for-Go2W基座](https://github.com/TrackinBIT/HIMLoco-for-Go2W)，来源提交见`UPSTREAM.json`。
- 保留原有BSD-3-Clause许可证、rsl_rl及资产声明；NVIDIA Isaac Gym软件不在包内。
>>>>>>> 6dbb972 (Add MUJICA Isaac Lab PPO terrain training)
