# 基于深度强化学习的云-边-端异构计算资源管理调度框架

本项目把任务 DAG、云/边/端异构资源、合法动作约束、调度策略接口、训练流程和评测指标拆成独立模块，评测脚本可以在同一组验证场景上统一比较 RL、HEFT、rollout 增强启发式、经典启发式、随机策略或用户自定义策略。

实现仅依赖 Python 标准库，便于在 openEuler、openKylin、OpenHarmony 兼容 Linux 环境及其他主流 Linux 发行版上运行。默认实现包含一个纯 Python 两层 MLP Actor，通过 rollout 增强 HEFT 行为克隆预热和 REINFORCE 策略梯度训练。训练阶段只在合法动作集合上采样，推理阶段采用 `rollout_safe` 模式：RL 给候选动作打分，同时用 HEFT 补全评估候选动作并保留 rollout 教师动作作为安全候选，从而降低分布外贪心决策风险。

## 一键运行

```bash
python run_all.py --config configs/default.json
```

该命令会自动完成：

1. 生成训练集与验证集场景；
2. 使用 rollout 增强 HEFT 行为克隆预训练 RL 策略；
3. 使用策略梯度继续训练；
4. 在验证集上评测 `heft`、`lookahead_heft`、`portfolio`、`max_min`、`random`、`rl`；
5. 输出 `results/summary.json` 和 `results/details.csv`。

快速自检可运行：

```bash
python tests/smoke_test.py
```

Linux 环境也可直接运行：

```bash
bash scripts/run_linux.sh
```

复杂场景泛化验证可运行：

```bash
python scripts/evaluate_complex.py
```

## 输出说明

- `artifacts/scenarios/train.json`：训练场景集。
- `artifacts/scenarios/val.json`：验证场景集。
- `artifacts/models/rl_policy.json`：训练得到的 RL 策略参数。
- `artifacts/training_log.json`：训练曲线与关键超参。
- `results/details.csv`：每个场景、每个策略的 makespan 和相对 HEFT 比值。
- `results/summary.json`：聚合指标，包括 `mean_makespan`、`std_makespan`、`mean_ratio`、`std_ratio` 和样本数。
- `results/complex/summary.json`：复杂 DAG 与更多资源条件下的泛化验证汇总。

## 赛题要求对齐

- 功能完整性：`run_all.py` 一键完成训练、验证评测和结果文件生成。
- 模块化：`cloud_edge_drl/domain.py`、`scenario.py`、`env.py`、`policies/`、`train.py`、`evaluate.py` 分别负责数据、场景、环境、策略、训练、评测。
- 插拔接口：所有策略实现 `Scheduler.choose_action(env)`；内置 HEFT、lookahead HEFT、portfolio、max-min、随机策略、RL 策略，并支持 `module:ClassName` 形式加载自定义策略。
- 合法动作处理：环境只暴露 ready task 与资源组合；RL softmax 只在 legal actions 上归一化，非法动作在 `env.step` 中会被拒绝。
- 性能指标：验证阶段输出 `mean_ratio = mean(policy_makespan / HEFT_makespan)`，并给出标准差与样本规模。
- 性能优化：默认 `rl` 采用学习策略 + rollout 安全校验，在默认验证集和复杂验证集上均达到低于 HEFT 的 `mean_ratio`。
- 泛化能力：配置文件把训练集和验证集分开生成，随机种子固定；`configs/complex.json` 提供更大 DAG、更密依赖和更多资源的未训练复杂验证集。
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
