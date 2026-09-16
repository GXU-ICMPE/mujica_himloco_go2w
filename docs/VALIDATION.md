# 交付验证记录

## 2026-09-16：X5loco v8 机器人及奖励迁移

已修改 X5 资源、具名关节接口、控制/停车 PI、三技能共用 v8 奖励、269 维 critic、
训练/续训/导出元数据和文档。仅阅读源码和改动内容。
按用户要求，**没有运行测试、语法编译、配置检查、物理仿真、PPO 或训练**。
下方所有通过数量均为旧版本记录，不能作为本次 X5 迁移的通过证据。
适配范围及留给用户的命令见 [X5_MIGRATION_ZH.md](X5_MIGRATION_ZH.md)。

## 2026-09-15：共用奖励姿态适配及原文审查

修改已完成，用户明确要求后续验证自行运行。

- 新增共用 `mujica/rewards.py`，复杂地形四项姿态权重、停车条件、局部高度参考及容忍区。
- S1 增加按任务的实际分项回报、未截断总回报和截零比例；S2 仍仅计算两项速度跟踪。
- 环境快照及元数据保存奖励配置，新增奖励续训兼容性检查。
- 用户要求到达前已启动的测试结果：**69 passed, 1 failed**。
- 失败项是 `test_both_backends_dispatch_to_the_shared_reward_implementation`：Gym 导入隔离还原 `sys.modules` 后，同源函数可能具有不同对象标识；已将对象同一性断言改为检查来源模块、方法名及源码路径。
- 上述断言修改后**未重跑**。没有继续执行测试、编译、配置检查、物理仿真或训练。
- 查阅原文及 Fig.2，并阅读当前模型、估计器、PPO、runner、随机化和电机代码；结论见 [PAPER_AUDIT_ZH.md](PAPER_AUDIT_ZH.md)。这属于源码审查。

以下较早的 57/43 项通过记录只对应修改前版本。

## 2026-09-15：新三类地形任务适配

任务已改为 `flat_slope`（平地及上下坡）、`discrete`（离散障碍）、`stairs`（上下楼梯）。
这是同日 Isaac Lab 源码迁移之后的独立改动，适配范围见[TERRAIN_TASKS_ZH.md](TERRAIN_TASKS_ZH.md)。

使用本机 `/home/xgy/miniforge3/envs/x3w_isaaclab/bin/python`，已执行：

- `python -m pytest -q tests`：**57 passed**。更新旧任务测试，新增地形任务覆盖测试。
- 六类地形的方向、递增难度、规则踏步、离散随机高度和平整出生区检查。
- 3/9/64/512 环境的任务均衡分配；至少 9 环境覆盖六种地形；plane 只标记为 `flat_slope`。
- S1/S2 的正立重置、全部技能立即驱动、20 秒上限、碰撞终止、越界截断、碰撞优先级。
- 课程须实际移动，失败降级，纯平地固定最低等级；升级不改变任务列。
- 旧技能检查点在续训、S2 初始化、导出、Sim2Sim 校验中被拒绝；新技能 CLI 和导出元数据正确。
- `python -m mujica.train --check-config --num-envs 64`：三类任务分配为 22/21/21；
  六种地形分配为 8/7/7/21/11/10，默认地形 20×18。
- 与修改前备份比较：`models.py`、`ppo.py`、`storage.py`、`motor.py` 字节未变；
  奖励函数语法树和完整 reward 配置未变。S1 仍为原 Go2W 奖励，S2 仍为两项速度跟踪。
- Python 语法编译、训练/导出/Sim2Sim/物理检查脚本的 CLI 帮助通过；六类高度场的离线预览已生成并检查。

新增测试运行的是生产逻辑的 CPU 张量和 NumPy 高度场；未启动 Isaac Sim/Gym、运行真实物理步或仿真 PPO。
测试通过不能证明轮式运动、不规则抬腿、规律抬腿已经学会。奖励设计留待下一步。

## 2026-09-15：本地 Isaac Lab 初次迁移验证（任务改版前）

本次对象是当前 `MUJICA_HIMLoco_Go2W_PPO/mujica_himloco_go2w` 目录。
使用 `/home/xgy/miniforge3/envs/x3w_isaaclab/bin/python`，读取安装版本：Python 3.11、
PyTorch 2.7.0+cu128、NumPy 1.26.0、SciPy 1.15.3、Isaac Lab Python 包 0.54.4、Isaac Sim 5.1.0.0。

已执行：

- 迁移前原测试：`28 passed`。
- 迁移后 `python -m pytest -q tests`：**43 passed**，包含 15 项新迁移测试。
- 新测试覆盖：33→19 刚体合并前后总质量/COM/惯量、碰撞数量、轴/限位、关节重排、wxyz 数学、地形组与坑深、58/270 观测、奖励逐项对照、恢复被动期、选择器切换、真正终态与单次历史更新、runner 适配、跨后端检查点拒绝。
- `python -m mujica.train --check-config`：离线检查通过，16 关节、19 刚体、19.523kg、5ms/20ms、348/270/16 维接口。
- Python 语法编译及新训练入口帮助检查通过。

测试里的算法优化器更新使用 CPU 合成张量。新环境生命周期测试排除 Isaac 绑定，运行生产方法；它们不创建 USD 场景，不执行物理步。

**本次未执行 Isaac Sim 启动、USD 导入、PhysX 接触/GPU rollout、仿真 PPO 更新、MuJoCo 或运动能力验证。**
新增的 `python -m mujica.isaaclab.sim_smoke` 是后续仿真验证入口，本次没有运行它。
迁移说明与后续命令见 [ISAACLAB_MIGRATION_ZH.md](ISAACLAB_MIGRATION_ZH.md)。

## 原始交付记录（迁移前，保留作来源说明）

日期：2026-09-15。这里区分算法代码验证、仿真接口验证与运动能力验证。验证环境：Python 3.12.14、PyTorch 2.14.0+cpu、NumPy 2.3.5、MuJoCo 3.13.0；不是用于Isaac Gym训练的目标旧版本环境。

## 已执行

- `python -m pytest -q tests`：28项测试通过。
- `python -m mujica.smoke`：8个张量环境，S1两轮更新，检查点续训至第3轮，S2一轮更新；整个底层actor和estimator参数保持不变。
- TorchScript：S1和S2导出、重新加载，与原模型在多步GRU及部分环境reset后的输出一致；测试中动作/状态最大绝对误差均为0。
- 环境生产方法的CPU契约测试：58/270维观测、真实碰撞分组、轮状态不被改写、最新帧技能更新、S1/S2终止规则、重置前的真正终态保存。
- 回报数据测试：超时用真正终态V(s_T)补偿；一般终止切断GAE；估计目标对应当前状态，SwAV目标对应下一状态；mini-batch覆盖余数。
- 电机测试：速度包络、-π/2附近小腿力矩峰值、校准上限覆盖、禁用后静态限幅、非法名称和速度检查。
- MuJoCo：实际加载包内Go2W模型（nq=23、nv=22、nu=16），通过自动技能模式站立初始姿态及手动Recovery仰躺初始姿态各1秒的headless运行；无非有限状态错误。
- Python语法编译、新训练CLI帮助及Git差异格式检查通过。

这些MuJoCo运行使用的是**测试用微型随机初始化网络**，验证加载、映射、推理、控制与物理仿真链路；没有证明它会行走或翻身。测试权重不随本源码包交付。

## 未执行

- Isaac Gym/PhysX真实环境启动和GPU rollout。当前验证机器没有安装Isaac Gym；CPU环境契约测试用了导入桩，不能代替PhysX验证。
- S1 30,000轮、S2 10,000轮的完整训练。
- 三技能收敛、最高登台高度、Recovery成功率、混合地形连续完成率。
- 真机电机校准、时延标定或sim2real。
- Isaac Gym目标旧PyTorch版本上的实际导出运行；当前CPU验证使用较新PyTorch，接口按1.10及以上设计。

交付包含全项目源码和机器人资源，但不包含NVIDIA Isaac Gym软件或已训练运动权重。

## 复现结构与实验结果的区别

本次PPO替换和Go2W奖励继承是用户明确要求。论文未公开的网络宽度、GRU训练截断、接触阈值、S2重置等细节已补全并标注。不能将这些工程选择称为作者原始实现，不能将测试通过称为论文实验结果复现。

以上为原交付历史记录。当前新任务和默认 Isaac Lab 后端的命令以主 README 为准。
