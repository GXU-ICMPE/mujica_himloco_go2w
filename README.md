# MUJICA · X5 / Go2W · Isaac Lab · PPO

在 MUJICA 两阶段复现框架中接入 **X5loco v8 的 X5 机器人、控制参数和奖励**。
新建 Isaac Lab 训练默认使用 X5，也可以显式传 `--robot x5`。
Go2W 保留为 `--robot go2w`；目录名和 Python 包名为兼容原环境保持不变。

**本次只完成源码适配，按用户要求没有执行测试、配置检查、仿真或训练。**
训练效果、接触稳定性及 4090 上的显存占用需要实际运行确认。

## 保留的 MUJICA 框架

- S1：三个地形技能，共享底层 actor、GRU 状态估计器，使用普通 PPO。
- S2：冻结 S1 网络，训练自动技能选择器；奖励只包含平面速度与偏航速度跟踪。
- actor 历史输入 `348 = 6 × 58`；X5 critic 为 269 维，Go2W 为 270 维。
- 正立重置、20 秒回合、地形课程、终态 bootstrap、技能协议沿用当前 MUJICA 实现。

| 技能 | 地形 | 目标 |
|---|---|---|
| `flat_slope` / 0 | 平地、上坡、下坡 | 以轮式行走为主 |
| `discrete` / 1 | 离散障碍 | 不规则抬腿越障 |
| `stairs` / 2 | 上楼、下楼 | 较规律地抬腿通过 |

三个技能共用 X5 v8 奖励。姿态容差随局部地形和运动指令变化，不依赖 S2 选择的技能。
这些运动方式是训练目标，没有在代码中强制固定步态。

## S1 训练

使用已安装 Isaac Sim / Isaac Lab 的 Python 环境。在服务器执行：

```bash
conda activate x3w_isaaclab
cd /root/autodl-tmp/mujica_himloco_go2w
python -m pip install -e . --no-deps
export OMNI_KIT_ALLOW_ROOT=1
bash scripts/train_x5_s1.sh
```

等价的完整训练命令：

```bash
python -m mujica.train \
  --backend isaaclab --robot x5 --stage s1 \
  --num-envs 512 --headless --seed 42 \
  --config configs/mujica_default.json \
  --iterations 30000
```

默认日志在 `logs/mujica_lab_x5_s1/<时间戳>/`，随代码位于数据盘。
512 是参考启动规模，未经本项目性能验证；需要减小规模时使用
`bash scripts/train_x5_s1.sh --num-envs 256`。脚本会把额外参数传给训练入口。
每轮 48 步，每步策略时间 20 ms（8 × 2.5 ms 物理步）。

**必须新训 X5 S1**：Go2W 的关节顺序、动力学和估计器维度不同；X5loco v8 的网络和观测结构
也与 MUJICA 不同。两种旧权重都不能直接作为这个 X5 任务的续训权重。

## 续训、S2 和播放

将以下 `<S1目录>` / `<S2目录>` 替换成真实训练日志目录：

```bash
# 续训恢复检查点中的机器人、奖励配置、优化器和迭代数；新增 5000 轮
python -m mujica.train --stage s1 --robot x5 \
  --resume <S1目录>/last.pt --num-envs 512 --headless --iterations 5000

# S1 技能具备所需能力后启动 S2
python -m mujica.train --stage s2 --robot x5 \
  --low-level <S1目录>/last.pt --num-envs 512 --headless --iterations 10000

# 图形播放 S1 的一个技能
python -m mujica.train --stage s1 --robot x5 \
  --resume <S1目录>/last.pt --num-envs 12 --play --skill flat_slope

# 图形播放 S2 的自动选择器
python -m mujica.train --stage s2 --robot x5 \
  --resume <S2目录>/last.pt --num-envs 12 --play --skill auto

# 导出 actor、GRU 和控制接口元数据
python -m mujica.export --checkpoint <S2目录>/last.pt --output exports/mujica_x5.pt
```

周期保存间隔为 1000 轮，正常结束时写 `last.pt`；中断后可续训最近的 `model_XXXXXX.pt`。
`--iterations` 始终表示本次新增轮数。续训会重置物理场景，不是某一物理帧的精确重放。
X5 TorchScript 导出保留显式 GRU 状态；**现有 MuJoCo 入口仍只支持 Go2W**，会拒绝 X5 元数据。
X5 的 MuJoCo 场景及对应停车 PI 控制器尚未迁移。

## 参数和文档

| 文件 | 内容 |
|---|---|
| `mujica/isaaclab/x5_config.py` | X5 控制、随机化、速度范围及 23 项奖励权重 |
| `mujica/isaaclab/x5_parameters.py` | v8 站姿、几何参考、姿态容差、抬腿落地条件 |
| `mujica/isaaclab/x5_state.py` | 局部地形分类、姿态代价、完整抬腿事件 |
| `mujica/isaaclab/x5_task.py` | 显式 PD、停车 PI、奖励和指令采样 |
| `mujica/isaaclab/robots.py` | X5 / Go2W 具名关节与碰撞观测接口 |
| `resources/robots/x5/` | 随仓库携带的 X5 URDF 和网格 |
| `mujica/models.py`、`ppo.py`、`storage.py` | MUJICA 网络与普通 PPO |

详细奖励、与 X5loco v8 的差异、后续自行检查命令见 [X5 迁移说明](docs/X5_MIGRATION_ZH.md)。
Go2W 分支见 [Go2W 原训练说明](docs/GO2W_TRAINING_ZH.md)。
较早的算法审查见 [MUJICA 论文结构审查](docs/PAPER_AUDIT_ZH.md)，其中机器人参数描述对应迁移前的 Go2W。
实际执行记录见 [验证记录](docs/VALIDATION.md)。

## 来源与许可

MUJICA 是独立结构复现，普通 PPO 替代 P3O，使用工程适配后的三类地形技能。
保留原有 HIMLoco / Go2W BSD-3-Clause 声明。
X5loco 参考来源、Apache-2.0 许可和本次修改范围见 [来源说明](third_party/X5loco-NOTICE.md)。
X5 资源来源记录见 `resources/robots/x5/source_manifest.json`。
