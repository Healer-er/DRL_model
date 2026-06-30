# 基于深度强化学习的云-边-端异构计算资源管理调度框架

本项目把任务 DAG、云/边/端异构资源、合法动作约束、调度策略接口、训练流程和评测指标拆成独立模块，评测脚本可以在同一组验证场景上统一比较 RL、HEFT、随机策略或用户自定义策略。

实现仅依赖 Python 标准库，便于在 openEuler、openKylin、OpenHarmony 兼容 Linux 环境及其他主流 Linux 发行版上运行。默认实现包含一个纯 Python 两层 MLP Actor，通过 HEFT 行为克隆预热和 REINFORCE 策略梯度训练，推理和训练阶段都只在合法动作集合上做选择。

## 一键运行

```bash
python run_all.py --config configs/default.json
```

该命令会自动完成：

1. 生成训练集与验证集场景；
2. 使用 HEFT 行为克隆预训练 RL 策略；
3. 使用策略梯度继续训练；
4. 在验证集上评测 `heft`、`random`、`rl`；
5. 输出 `results/summary.json` 和 `results/details.csv`。

快速自检可运行：

```bash
python tests/smoke_test.py
```

Linux 环境也可直接运行：

```bash
bash scripts/run_linux.sh
```

## 输出说明

- `artifacts/scenarios/train.json`：训练场景集。
- `artifacts/scenarios/val.json`：验证场景集。
- `artifacts/models/rl_policy.json`：训练得到的 RL 策略参数。
- `artifacts/training_log.json`：训练曲线与关键超参。
- `results/details.csv`：每个场景、每个策略的 makespan 和相对 HEFT 比值。
- `results/summary.json`：聚合指标，包括 `mean_makespan`、`std_makespan`、`mean_ratio`、`std_ratio` 和样本数。

## 赛题要求对齐

- 功能完整性：`run_all.py` 一键完成训练、验证评测和结果文件生成。
- 模块化：`cloud_edge_drl/domain.py`、`scenario.py`、`env.py`、`policies/`、`train.py`、`evaluate.py` 分别负责数据、场景、环境、策略、训练、评测。
- 插拔接口：所有策略实现 `Scheduler.choose_action(env)`；内置 HEFT、随机策略、RL 策略，并支持 `module:ClassName` 形式加载自定义策略。
- 合法动作处理：环境只暴露 ready task 与资源组合；RL softmax 只在 legal actions 上归一化，非法动作在 `env.step` 中会被拒绝。
- 性能指标：验证阶段输出 `mean_ratio = mean(policy_makespan / HEFT_makespan)`，并给出标准差与样本规模。
- 泛化能力：配置文件把训练集和验证集分开生成，随机种子固定，便于复现和替换划分。
- 文档质量：`docs/DESIGN.md` 解释建模、动作、奖励、训练和评估；`docs/FILE_OVERVIEW.md` 对每个文件做了详细说明。

## 自定义策略

自定义策略只需继承或遵守 `Scheduler` 接口：

```python
class MyScheduler:
    name = "my_scheduler"

    def reset(self, scenario):
        pass

    def choose_action(self, env):
        return env.legal_actions()[0]
```

然后在评测时使用：

```bash
python -m cloud_edge_drl.cli evaluate --config configs/default.json --policies heft,examples.custom_policy:FastestReadyScheduler
```

## 目录说明

详细文件级说明见 `docs/FILE_OVERVIEW.md`。
