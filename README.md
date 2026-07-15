# 基于深度强化学习的云-边-端异构资源调度框架

本项目实现了一个纯 Python 的云-边-端任务调度实验框架，用于在同一组 DAG 任务场景上比较强化学习调度器、HEFT 系列启发式、经典启发式、随机策略和用户自定义策略。

框架将任务 DAG、异构计算资源、通信代价、合法动作约束、策略接口、训练流程和评测指标拆分为独立模块。默认实现包含一个两层 MLP Actor，通过 lookahead HEFT 教师策略进行行为克隆预训练，再使用 REINFORCE 继续优化。RL 推理默认使用 `rollout_safe` 模式：模型为候选动作打分，同时由 rollout 教师评估候选动作并保留安全候选，从而降低贪心决策的分布外风险。

项目不依赖第三方 Python 包，推荐使用 Python 3.8 及以上版本。

## 快速开始

```bash
python run_all.py --config configs/default.json
```

该命令会按顺序完成：

1. 生成训练集和验证集场景；
2. 训练 RL 策略，并保存行为克隆阶段模型；
3. 在验证集上评测默认策略列表；
4. 输出模型、训练日志和评测结果。

也可以使用模块入口：

```bash
python -m cloud_edge_drl.cli run-all --config configs/default.json
```

快速自检：

```bash
python tests/smoke_test.py
```

运行环境正确性单元测试：

```bash
python -m unittest tests.test_env_correctness
```

Windows 和 Linux 脚本入口：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_windows.ps1
```

```bash
bash scripts/run_linux.sh
```

## CLI 用法

主入口 `cloud_edge_drl.cli` 支持以下子命令；不指定子命令时默认执行 `run-all`。

```bash
python -m cloud_edge_drl.cli generate --config configs/default.json
python -m cloud_edge_drl.cli train --config configs/default.json
python -m cloud_edge_drl.cli evaluate --config configs/default.json
python -m cloud_edge_drl.cli run-all --config configs/default.json
```

常用参数：

- `--config`：指定 JSON 配置文件，默认 `configs/default.json`。
- `--quick`：运行更小规模的快速实验，适合本地冒烟测试。
- `evaluate --policies`：用逗号分隔指定评测策略。
- `evaluate --model`：指定 RL 模型路径。

示例：

```bash
python -m cloud_edge_drl.cli evaluate --config configs/default.json --policies heft,lookahead_heft,rl
python -m cloud_edge_drl.cli run-all --config configs/default.json --quick
```

## 内置策略

默认评测策略来自 `configs/default.json`：

- `heft`：基础 HEFT 调度器。
- `lookahead_heft`：带一步前瞻/rollout 增强的 HEFT。
- `portfolio`：组合式启发式策略。
- `min_min`、`max_min`、`mct`、`met`、`olb`：经典调度启发式。
- `random`：随机合法动作策略。
- `rl_bc_only`：仅使用行为克隆模型的 RL 策略。
- `rl_greedy`：使用训练后模型直接贪心推理。
- `rl`：默认安全 rollout 推理模式。

## 配置文件

- `configs/default.json`：默认训练和验证设置。默认生成 24 个训练场景、10 个验证场景，每个 DAG 包含 8 到 14 个任务，资源包含 1 个 cloud、2 个 edge、2 个 device。
- `configs/complex.json`：更复杂的泛化验证设置。默认生成 32 个训练场景、12 个验证场景，每个 DAG 包含 18 到 30 个任务，资源包含 2 个 cloud、4 个 edge、4 个 device。

配置中主要包含：

- `paths`：场景、模型、日志和结果目录。
- `scenario`：DAG 数量、任务规模、依赖边概率、计算量和数据量范围。
- `resources`：cloud、edge、device 数量、速度范围和跨资源带宽。
- `training`：训练轮数、行为克隆轮数、教师策略、推理模式、隐藏层大小和学习率。
- `evaluation`：默认评测策略列表。

## 输出文件

默认运行会生成或更新以下文件：

- `artifacts/scenarios/train.json`：训练场景集。
- `artifacts/scenarios/val.json`：验证场景集。
- `artifacts/models/rl_policy.json`：完整训练后的 RL 策略参数。
- `artifacts/models/rl_policy_bc_only.json`：行为克隆阶段策略参数。
- `artifacts/training_log.json`：训练日志和关键指标。
- `results/details.csv`：每个场景、每个策略的 makespan、HEFT makespan 和相对 HEFT 比值。
- `results/summary.json`：聚合评测指标，包括 `mean_makespan`、`std_makespan`、`mean_ratio`、`std_ratio` 和样本数。

复杂场景评测会输出到 `results/complex/`。

## 复杂场景与泛化评测

使用默认配置训练，并在复杂 held-out 场景上评测：

```bash
python scripts/evaluate_complex.py
```

运行多随机种子的泛化基准：

```bash
python scripts/benchmark_generalization.py --config configs/default.json --seeds 2026,2027,2028
```

可选缩小规模：

```bash
python scripts/benchmark_generalization.py --train-count 8 --val-count 4 --episodes 20
```

该脚本会在 `results/generalization/` 下写入每个 seed 的结果，并生成：

- `aggregate.json`：跨 seed 聚合数据。
- `report.md`：按 `mean_ratio` 排序的 Markdown 报告。

## 可视化报告

生成 Markdown 报告和 SVG 甘特图：

```bash
python scripts/generate_visual_report.py --config configs/default.json
```

默认输出目录为 `results/visual_report/`，包含：

- `report.md`：策略排序和甘特图索引。
- `summary.json`：聚合指标。
- `details.csv`：逐场景评测明细。
- `timelines.json`：调度时间线。
- `gantt_*.svg`：指定场景下各策略的甘特图。

也可以指定策略、模型、场景或输出目录：

```bash
python scripts/generate_visual_report.py --policies heft,lookahead_heft,rl --scenario-id val_003
```

## 自定义策略

自定义策略需要实现 `choose_action(env)`，并可选实现 `reset(scenario)`。`choose_action` 必须返回一个合法动作 `(task_id, resource_id)`。

```python
class MyScheduler:
    name = "my_scheduler"

    def reset(self, scenario):
        pass

    def choose_action(self, env):
        return env.legal_actions()[0]
```

评测时可以用 `module:ClassName` 或 `path/to/file.py:ClassName` 加载：

```bash
python -m cloud_edge_drl.cli evaluate --config configs/default.json --policies heft,examples.custom_policy:FastestReadyScheduler
```

## 项目结构

```text
cloud_edge_drl/
  cli.py                  # 命令行入口
  config.py               # 配置读取、quick 模式和目录初始化
  domain.py               # Task、Edge、Resource、Scenario 等领域对象
  env.py                  # 调度环境、合法动作和时间估计
  scenario.py             # 场景生成、保存和读取
  train.py                # RL 训练流程
  evaluate.py             # 策略评测流程
  metrics.py              # 指标聚合和排序
  policies/               # 内置策略和策略注册表
configs/                  # 默认与复杂场景配置
scripts/                  # 批量评测、可视化和跨 seed 基准脚本
tests/                    # 冒烟测试与环境正确性测试
artifacts/                # 生成的场景、模型和训练日志
results/                  # 评测结果和报告
docs/OPTIMIZATION_LOG.md  # 优化记录
```

## 评测指标

核心指标为：

```text
ratio_to_heft = policy_makespan / HEFT_makespan
```

`mean_ratio < 1.0` 表示该策略平均优于 HEFT，`mean_ratio = 1.0` 表示与 HEFT 持平。所有策略都通过同一个 `SchedulingEnv` 调度，环境只暴露当前 ready task 与资源组合，非法动作会在 `env.step` 中被拒绝。
