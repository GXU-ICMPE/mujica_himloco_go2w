# MUJICA 当前状态与 Isaac Lab 迁移

> 本文的机器人参数、维度和奖励描述对应原 Go2W 分支。2026-09-16 新增 X5 默认训练入口，详见 [X5 迁移说明](X5_MIGRATION_ZH.md)。

日期：2026-09-15。检查对象是本目录 `MUJICA_HIMLoco_Go2W_PPO/mujica_himloco_go2w`。

## 结论

当前项目已有 MUJICA 两阶段结构的 PPO 实现，包含共享三技能 actor、GRU 状态估计、SwAV、S2 自动选择器、续训和导出。本次将默认仿真入口迁到 Isaac Lab，保留算法和 Go2W 奖励。代码迁移与离线验证完成；真实 Isaac Lab 场景和训练尚未执行。

本目录未发现训练检查点或训练指标文件。原 `docs/VALIDATION.md` 的 MuJoCo 记录属于此前交付，不是本次 Lab 后端的运行证据。当前目录没有 Git 元数据；本次没有提交。

## 复现到哪一层

| 层次 | 当前证据与限制 |
|---|---|
| S1 三技能共享策略 | `models.py` + `runner.py` + 环境任务分配；`flat_slope` / `discrete` / `stairs` 同批训练 |
| 状态估计 | 6 帧 MLP + 持续 GRU；速度、18 组碰撞、4 轮离地距离、latent；SwAV 下一状态监督 |
| S2 选择器 | 冻结底层 actor 和 estimator，离散 PPO 选择技能，每个 20ms 控制步选择一次 |
| 回报时间语义 | 真正终态用于超时 V(s_T) 和下一帧监督；done 环境清空 GRU，不把新回合观测当旧回合终态 |
| 算法与奖励偏离 | 项目原约定已经将 P3O 改为 PPO，S1 三技能共用 Go2W 15 项奖励；未做约束 critic/P3O 罚项 |
| 训练效果 | 无本项目收敛证据；CPU 接口测试不能证明平地/坡道、离散障碍、楼梯运动能力 |
| 论文实验结果 | 未验证成功率、登台高度、综合地形任务或真机结果 |

框架迁移后，任务已按用户要求改为平地及上下坡、离散障碍、上下楼梯三类。
本轮已在共用奖励中放宽复杂地形的四项姿态约束，详见[奖励说明](REWARDS_ZH.md)。具体变更见[任务说明](TERRAIN_TASKS_ZH.md)。

## 迁移后的数据链

```text
python -m mujica.train  [默认 --backend isaaclab]
  → AppLauncher 启动 Isaac Sim
  → MUJICAEnv(DirectRLEnv)
      本地 URDF → 固定附件合并 → Lab USD 导入 → Articulation
      MUJICA 地形高度场 → Lab 三角网格
      每 5ms 显式混合 PD + 电机包络 → 关节力矩
      每 20ms 状态/接触 → 奖励/终止 → 保存终态 → 重置 → 观测
  → MUJICAVecEnv 将 Lab 五返回值转为原 runner 七返回值
  → 原 MUJICARunner / PPO / Storage / Models
  → 原 export / MuJoCo sim2sim
```

使用 Direct 工作流是为了保持现有每步控制和训练组织；可参照
[Isaac Lab 官方迁移说明](https://isaac-sim.github.io/IsaacLab/main/source/migration/migrating_from_isaacgymenvs.html)。
具体 API 已按本机 `/home/xgy/IsaacLab/source/isaaclab/isaaclab` 核对，尚未启动应用验证。

## 当前接口与任务约定

- 58 维单帧 × 6 帧 = 348 维 actor 输入；最新帧在前。
- critic 270 维 = 本体 58 + 速度 3 + 碰撞 18 + 轮地距离 4 + 高度 187。
- 16 维动作，12 个腿关节位置偏移 + 4 个轮速度目标。
- 5ms 物理步、4 倍降采样、20ms 策略步；三类任务均从第一步开始驱动。
- S1 共用 15 个 Go2W 奖励项，复杂地形覆盖四项姿态权重；保留 dt 缩放、正截断，S2 两项速度跟踪。
- 默认 20 个课程等级、18 列地形，六种地形重复三组；三类任务单独均衡分配环境。
- S1/S2 最长 20s、正立重置；机身碰撞终止，接近地形块边界截断；无随机翻倒恢复任务。
- 原命令范围、噪声、随机 Kp/Kd、电机强度、摩擦、基体质量、COM、动作延迟和扰动设置保留；课程与日志按新任务更新。

参数快照在 `mujica/isaaclab/defaults.json`，与本包 Gym 配置继承链同步；任务字段已更新为 v2，未使用其他同名 MUJICA 项目的配置。

## 不能直接照搬的接口

### 关节和四元数

策略的 16 维顺序明确规定为 `FL, FR, RL, RR`，每腿 `hip, thigh, calf, foot`。
Lab 关节按名字查找后 gather/scatter；不能假定其内部顺序与旧 Gym 一致。
导出 JSON 同时记录策略顺序、Lab 实际顺序和映射索引。

Lab 内部使用 `wxyz` 四元数。旧配置的初始 `xyzw` 在配置构造时转换。
轮关节角度只在观测副本中清零，仿真状态保持实际连续轮角。

### 固定附件与接触

原 URDF 有 33 个 link，其中包括固定小腿外壳、电机外壳、IMU、雷达和两个头部。
不能依赖 Lab 导入器与 Gym 对 `dont_collapse` 的解释相同：

1. `assets.py` 预先合并固定附件，转换几何、质量、质心、惯量和下游 joint origin。
2. 保留 `Head_upper`、`Head_lower`，得到 19 个刚体；Lab 导入时禁用进一步合并。
3. `ContactSensor` 按名称读取 19 个真实刚体，两个头部合成一组，形成 18 组标签。
4. 验证实际导入刚体/关节名称；名称不符时立即报错。

离线确认合并前后总质量均约 19.523kg，世界坐标下总 COM、总惯量、碰撞数量和活动轴/限位保持一致。USD、PhysX 接触和约束尚未验证。

生成 URDF/USD 缓存在 `.cache/isaaclab/`；原始资产保持不变。缓存含绝对网格路径，移动项目或更改网格后应删除缓存再生成。

### 重置和训练监督

Lab 默认在返回观测前自动重置。新环境在 `_get_rewards()` 中保存真正终态，再让 Lab `_reset_idx()` 重置。
普通环境每步只追加一帧；重置环境将新首帧复制六次。超时与一般终止分开，机身碰撞和超时同一帧发生时按一般终止处理。
重置时用 URDF 正运动学恢复轮心位置，不额外推进物理时间，也不复用上回合轮心位置。

### 电机、地形和随机化

- Lab 导入器和 implicit actuator 的内部 PD 均设为零，每个物理步由原显式 PD 与电机包络计算力矩。
- 轮地距离和 187 点高度沿用高度场采样与轮半径近似；未改成射线传感器，倾斜车轮仍有近似误差。
- 两个后端现在共用 `mujica/terrain.py` 的六类高度场。Lab 使用安装版本的高度场转三角网格工具；Gym 使用其自身工具，不声称两者网格与接触数值逐位一致。
- `disturbance_interval=8` 在旧源码中直接按**控制步**取模；新实现保持每 8 控制步的下一物理步短脉冲。`push_interval_s=15` 才是秒。
- 随机摩擦在 reset 通过 PhysX CPU 属性接口写入；大量同步重置的性能需实测。质量变化按比例更新惯量，PhysX 跨版本数值等价尚未证明。
- `--no-randomization` 关闭 domain-randomization 开关，保留关节/位置/初速度重置扰动；三类任务均使用默认正立朝向。

## 后续运行命令

在项目根目录，使用 `x3w_isaaclab` 环境。

```bash
# 已执行：离线检查，不启动 Isaac Sim
python -m mujica.train --check-config
python -m pytest -q tests

# 待执行：纯物理接口检查，零动作，无 PPO 更新
python -m mujica.isaaclab.sim_smoke --num-envs 9 --steps 350 --terrain plane --headless
python -m mujica.isaaclab.sim_smoke --num-envs 9 --steps 350 --terrain trimesh --headless
python -m mujica.isaaclab.sim_smoke --stage s2 --num-envs 9 --steps 350 --terrain trimesh --headless

# 待执行：小地形短训练；这不是正式基线
python -m mujica.train --stage s1 --num-envs 64 --terrain-rows 2 --terrain-cols 6 \
  --iterations 2 --headless --log-dir logs/terrain_posture_v1_startup_check

# 通过物理接口检查后，默认20×18地形，从新目录训练
python -m mujica.train --stage s1 --num-envs 512 --headless \
  --config configs/mujica_default.json --iterations 30000 --log-dir logs/terrain_posture_v1_s1

# S1技能分别验收后启动S2
python -m mujica.train --stage s2 --num-envs 512 --headless \
  --low-level logs/terrain_posture_v1_s1/last.pt --iterations 10000 --log-dir logs/terrain_posture_v1_s2
```

纯物理脚本仅检查加载、维度、有限值与终态接口；零动作不等于站立或运动验收。
512 是待实测的起步环境数，显存不足应下降；本次没有进行显存或吞吐量测试。

`--resume` 只接受同后端的 Lab 检查点，并保持保存的环境设置。
`--low-level` 可以用于新 S2 的底层初始化，但要求关节顺序和 v2 技能元数据完全一致；旧 Moving / Climb / Recovery 权重会被拒绝。
小地形检查点续训会继续沿用其小地形配置，正式基线应创建新日志目录重新训练。

## 验证边界

本次离线验证包含：原算法测试、新生产任务方法的张量测试、资源合并、关节顺序、四元数、奖励公式对照、终态/历史/选择器时序、CLI 和 Python 语法。
详见 [验证记录](VALIDATION.md)。

未执行 Isaac Sim 应用启动、USD 导入、真实 ContactSensor 读数、GPU rollout、仿真 PPO 更新、收敛或真机测试。
接下来先验收真实场景和三技能初始条件，再做训练和奖励实验。
