# 论文结构与代码对应

默认仿真后端已迁到 Isaac Lab，算法组织保持本文件描述。新环境路径是
`mujica/isaaclab/env.py`，观测及基础奖励在 `task_logic.py`，共用奖励适配在 `mujica/rewards.py`，任务生命周期在共享的 `mujica/locomotion_tasks.py`；
下面的 `mujica_robot` 是同步新任务语义的 Gym 后端。
2026-09-15 对照原文的审查见 [PAPER_AUDIT_ZH.md](PAPER_AUDIT_ZH.md)。
后端迁移与当前验证边界见 [迁移说明](ISAACLAB_MIGRATION_ZH.md) 和 [验证记录](VALIDATION.md)。

本项目以 [MUJICA v1](https://arxiv.org/abs/2605.13058) 为结构依据，基于
[TrackinBIT/HIMLoco-for-Go2W](https://github.com/TrackinBIT/HIMLoco-for-Go2W)
提交 `011693738c61603c3f22f2bce755098dd36fa7eb` 修改。

## 用户明确要求的适配

1. P3O → HIMLoco 风格 PPO：保留裁剪策略目标、GAE、裁剪价值损失、熵奖励、KL 自适应学习率；删除约束 critic、cost advantage、P3O 惩罚目标。
2. 当前三类地形任务共用 Go2W 的 15 项奖励。离散障碍和楼梯共用较宽松的四项姿态权重及局部高度/停车容忍区；dt 缩放和总奖励正截断保留。
3. 原任务集合改为平地及坡道、离散障碍、上下楼梯。没有新增步态或抬腿奖励。详情见[当前奖励](REWARDS_ZH.md)。

S2 使用论文要求的统一速度跟踪目标：复用 Go2W 的线速度及角速度跟踪函数与权重。它不使用任务标签或另外设计的选择器奖励。

## 确定的信号与维度

| 信号 | 维度/顺序 | 代码 |
|---|---|---|
| 单步本体观测 | ω3、g3、cmd3、q16、qdot16、上一动作16、技能标量1，共58 | `mujica_robot._current_frame` |
| 历史 | 最近6帧，最新帧在前，共348 | `compute_observations` |
| 特权观测 | o58、v3、c18、u4、h187，共270 | `_current_frame` |
| 估计器 | 历史MLP → 持续GRU状态 → v3、c18、u4、latent16 | `mujica/models.py:StateEstimator` |
| 底层actor | 当前o58 + 全部估计特征41，共99 → 16动作 | `MultiSkillActorCritic` |
| 底层critic | 特权270 → V标量 | `MultiSkillActorCritic.critic` |
| 选择器actor | 每帧去掉技能标量，6×57=342 → 3类分布 | `SelectorActorCritic` |
| 选择器critic | 特权270 → V标量 | `SelectorActorCritic.critic` |

技能编码为 `flat_slope=0`（平地与上下坡）、`discrete=1`（离散障碍）、`stairs=2`（上下楼梯）。
协议版本为 2；旧三技能检查点不能直接加载。地形分配、课程和重置见[任务说明](TERRAIN_TASKS_ZH.md)。
当前任务划分按用户的地形/抬腿需求调整，已经不同于论文原任务集合。编码数值是工程选择。轮关节位置在**观测副本**中置零，遵循基座的连续轮编码；不会写坏仿真状态。速度、命令和高度图沿用基座缩放；估计速度监督目标使用同样缩放。

## S1：联合训练

多任务环境在同一批 rollout 中提供三种任务，一个共享 actor 使用标量技能区分任务。所有技能共享 actor、estimator、reward critic 和优化器；没有专家蒸馏或 MoE。

估计器监督时间对齐为：

- `history_t, hidden_t → v_hat_t, c_hat_t, u_hat_t, latent_t`。
- 速度MSE、碰撞BCE、距离MSE对应**当前**真实状态。
- reference encoder 使用**真实下一帧** `o_(t+1)`；SwAV 对齐两个 latent。
- episode 自动重置时，使用重置前的终止帧；绝不把新一局初始帧作为上一局的预测目标。

GRU隐状态由 runner 在每个控制步保存、传递，只有对应环境 done 时清零。网络训练重放采样时保存的隐状态，采用一步截断 BPTT。论文没有公开序列训练细节，这一实现方式属于工程补全。

PPO 使用采样时冻结保存的估计特征进行本批策略更新；估计器由独立优化器监督更新。PPO 梯度不进入估计器，避免一个 rollout 内估计器的变化使动作概率基准失配。这是对基座更新组织方式的明确调整，PPO 裁剪目标本身保持一致。该KL只衡量给定采样特征后的actor变化，不能约束估计器更新引起的完整闭环策略变化；这与完整序列重放的端到端recurrent PPO不同。

## S2：冻结底层、训练选择器

从 S1 检查点读取整个底层网络，冻结 actor、estimator 及其 reference/prototypes。每个20ms控制步：

1. 选择器读取六帧、逐帧移除技能标量后的本体历史。
2. 从分类分布采样技能（评估使用argmax）。
3. 更新最新帧技能标量，输入同一个冻结底层actor。
4. 底层确定性动作送入同一个环境，训练选择器 PPO。

切换技能不会重建地形、改变命令、重置机器人、重置GRU或改动历史技能值。
S1/S2 对所有任务使用相同的正立重置与 20 秒时限；机身碰撞终止，接近地形块边界截断。
S2 的实际任务标签仅用于地形分配、课程和日志，不输入选择器 actor；critic 可使用特权信息。

## 执行器

12个腿关节动作是默认关节角偏移，4个轮关节动作是速度目标：

`tau_leg = Kp * (q_default + 0.25*a - q) - Kd*qdot`

`tau_wheel = Kd * (10*a - qdot)`

保留基座稳定负反馈；论文公式中的PD符号与图示有歧义，不能机械照抄为正反馈。动作顺序使用真实Isaac DOF名称，导出时写入JSON，MuJoCo按名称映射。

电机包络每个5ms物理步计算。速度低于拐点时力矩上限不变，高速线性下降；小腿再乘与角度余弦相关的因子。默认峰值和速度来自URDF，速度拐点及余弦系数没有实测校准。
`calibrated` 是元数据标记，不会自动使参数成为实测值。`violation_count` 仅监控，不进入 PPO 奖励/约束优化。

## 不能称为作者原始参数的部分

网络层宽、latent16、GRU64、prototype32、接触阈值1N、技能编码值、碰撞分组、轮地距离采样、GRU截断方式、当前三类地形课程阈值及重置分布，均在代码中明确给出，但原文没有披露全部精确实现。

Go2W原始URDF有33个link，按固定关节规则合并后保留19个刚体，本项目将两个物理头部刚体的接触合并为一个通道，得到18个真实碰撞通道。不是补零到18维。轮地距离使用轮心下方地形高度减去轮半径；倾斜车轮时这是近似量。

论文行为成绩还取决于奖励、约束、实测电机参数及长时间训练。按用户要求替换PPO和原Go2W奖励后，本项目交付的是**结构复现的可训练实现**，并不等同于论文实验结果复现。
