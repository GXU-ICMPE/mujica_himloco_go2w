# 备用 Isaac Gym 后端

当前默认入口是 Isaac Lab，见[主 README](../README.md)。Gym 后端保留用于已有环境中的对照；
`legged_gym/envs/mujica/` 已同步当前三类地形任务，不再提供旧 Moving / Climb / Recovery 训练入口。
原版 `go2w` 环境未修改。

## 环境与入口

使用已有 Isaac Gym Preview 4 / HIMLoco 环境。Isaac Gym 必须先于 torch 导入；Isaac Lab/Sim 不包含这个旧运行库。
Gym 软件本体不在项目包内。

```bash
python -c "import isaacgym; import torch; print(torch.__version__, torch.cuda.is_available())"
python -m pip install -e ./rsl_rl --no-deps
python -m pip install -e . --no-deps
python -m pip install -r requirements-isaacgym.txt

# 待运行的小规模启动检查
python -m mujica.train_isaacgym --stage s1 --num-envs 64 --headless --iterations 2 --log-dir logs/terrain_gym_startup_check

# S1；环境数量按实际显存调整
python -m mujica.train_isaacgym --stage s1 --num-envs 512 --headless --iterations 30000 --log-dir logs/terrain_gym_s1

# S1 三类技能验收后再训练 S2
python -m mujica.train_isaacgym --stage s2 --num-envs 512 --headless --low-level logs/terrain_gym_s1/last.pt --iterations 10000 --log-dir logs/terrain_gym_s2
```

也可以用 `python -m mujica.train --backend isaacgym ...` 显式选择此后端。
`--iterations` 是本次新增迭代数。本轮未执行 Gym 仿真或训练，上面不是已验收的性能配置。

## 任务与检查点

两个后端共用 `mujica/skills.py`、`terrain.py`、`locomotion_tasks.py`：

- `flat_slope=0`：平地和上下坡。
- `discrete=1`：随机离散障碍。
- `stairs=2`：上下楼梯。

默认地形为 20×18，三类任务的环境数量最多相差 1。重置、课程、终止、日志见[任务说明](TERRAIN_TASKS_ZH.md)。
S1 共用 Go2W 奖励项，并按实际任务应用四项姿态放宽、停车容忍区和局部高度参考；见[奖励说明](REWARDS_ZH.md)。

播放的 `--skill` 只接受 `auto`、`flat_slope`、`discrete`、`stairs`；S1 不支持 `auto`。
旧三技能检查点不能续训、供 S2 初始化或重新导出。新模型也应在原后端续训，不能把 Gym/Lab 物理状态等同。
导出和 MuJoCo 的新接口见[主 README](../README.md)。
