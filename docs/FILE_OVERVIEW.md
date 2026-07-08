# 文件详细说明

## 根目录文件

### `README.md`

项目入口说明文档。包含赛题要求对齐、一键运行命令、输出文件解释、自定义策略接入方式和目录说明。评审可以从该文件快速了解如何复现实验。

### `run_all.py`

一键运行入口。默认执行 `generate -> train -> evaluate` 全流程，生成训练集和验证集、训练 RL 模型，评测 HEFT、lookahead HEFT、portfolio、max-min、随机和 RL 策略，并产出 `results/summary.json` 与 `results/details.csv`。

### `requirements.txt`

依赖说明。当前实现不依赖第三方 Python 包，只要求 Python 3.8 或更高版本，降低在 openEuler、openKylin 等系统上的部署成本。

### `.gitignore`

忽略 Python 缓存、虚拟环境和临时文件，避免实验过程产生的中间文件污染版本管理。

## 配置

### `configs/default.json`

默认实验配置。包含随机种子、输出路径、任务 DAG 生成参数、云/边/端资源数量、计算速度范围、跨资源通信带宽、训练轮数、行为克隆轮数、学习率和默认评测策略。替换任务规模、资源类型与异构参数时优先修改该文件。

## 核心包 `cloud_edge_drl`

### `cloud_edge_drl/__init__.py`

包初始化文件，声明项目版本号。

### `cloud_edge_drl/domain.py`

定义核心数据结构：`Task`、`Edge`、`Resource`、`Scenario`。`Scenario` 负责维护任务父子关系、边数据量、资源字典、带宽矩阵校验和 JSON 序列化，是整个框架的数据基础。

### `cloud_edge_drl/config.py`

配置加载和运行模式工具。提供 `load_config()` 读取 JSON 配置，`ensure_output_dirs()` 创建输出目录，`apply_quick_mode()` 缩小实验规模用于快速自检。

### `cloud_edge_drl/utils.py`

通用工具函数。包含 JSON 读写、均值、方差、标准差、数值裁剪等小工具，供训练和评测模块复用。

### `cloud_edge_drl/scenario.py`

场景生成与加载模块。根据配置随机生成 DAG、任务计算量、边通信量、异构资源速度和资源间带宽矩阵；支持保存和加载 `train.json`、`val.json`，保证训练集与验证集划分可复现。

### `cloud_edge_drl/env.py`

调度环境模块。实现 ready task 约束、合法动作集合、legal mask、最早开始/结束时间计算、资源可用时间更新、makespan 计算和非法动作拒绝。训练与评测都通过该环境交互，确保行为一致。

### `cloud_edge_drl/features.py`

动作特征提取模块。为每个合法 `(task, resource)` 动作构造固定长度特征，包括任务计算量、父子数量、upward rank、资源速度、资源空闲时间、估计开始/结束时间、通信等待和资源类型 one-hot。RL 策略依赖该模块适配不同规模场景。

### `cloud_edge_drl/train.py`

RL 训练流程。先用 HEFT 进行行为克隆预热，再使用 masked REINFORCE 做策略梯度训练，保存模型参数和训练日志。训练目标使用 `-makespan / lower_bound`，并用滑动 baseline 降低方差。

### `cloud_edge_drl/evaluate.py`

统一评测流程。对每个策略和每个验证场景运行调度，计算 makespan、相对 HEFT 的 ratio、步数，并写出 CSV 明细和 JSON 汇总。所有策略共享同一环境和指标。

### `cloud_edge_drl/metrics.py`

指标聚合模块。按策略聚合 `mean_makespan`、`std_makespan`、`mean_ratio`、`std_ratio` 和样本数量，并提供按 ratio 排序的方法。

### `cloud_edge_drl/cli.py`

命令行入口。支持 `generate`、`train`、`evaluate`、`run-all` 子命令，也支持 `--quick` 快速模式和自定义策略列表，便于批量实验和评测脚本调用。

## 策略模块 `cloud_edge_drl/policies`

### `cloud_edge_drl/policies/__init__.py`

导出内置策略类，方便外部代码直接导入。

### `cloud_edge_drl/policies/base.py`

统一策略接口定义。所有调度器只需实现 `reset(scenario)` 和 `choose_action(env)`，评测脚本即可统一调用。

### `cloud_edge_drl/policies/heft.py`

HEFT 风格启发式基线。计算任务 upward rank，在每一步选择 rank 最高的 ready task，再选择能让该任务最早完成的资源。该策略作为外部调度方法和 ratio 指标基准。

### `cloud_edge_drl/policies/random_policy.py`

随机合法动作基线。从 `env.legal_actions()` 中随机选择动作，用于检查环境、mask 和评测流程是否正常，也可作为性能下界参考。

### `cloud_edge_drl/policies/lookahead_heft.py`

rollout 增强 HEFT 策略。枚举当前所有合法动作，先执行候选动作，再用 HEFT 补全剩余任务，选择最终 makespan 最低的动作。

### `cloud_edge_drl/policies/classic.py`

经典异构调度启发式集合。当前包含 `MinMinScheduler` 和 `MaxMinScheduler`，用于扩展赛题中的外部调度方法对比。

### `cloud_edge_drl/policies/portfolio.py`

组合策略。离线运行 HEFT、lookahead HEFT、min-min 和 max-min，选择完整计划 makespan 最低的策略轨迹执行。

### `cloud_edge_drl/policies/rl_policy.py`

纯 Python 两层 MLP Actor。对每个合法动作提取特征并打分，只在合法动作集合上做 softmax 与采样；支持行为克隆和 REINFORCE 更新，并能保存/加载 JSON 模型。默认推理使用 `rollout_safe` 模式，将 MLP 高分动作与 rollout 教师动作一起做 HEFT 补全校验，避免轻量模型在未见场景上做出明显差于 HEFT 的贪心选择。

### `cloud_edge_drl/policies/registry.py`

策略工厂。根据字符串创建 `heft`、`lookahead_heft`、`portfolio`、`min_min`、`max_min`、`random`、`rl` 或 `module:ClassName` 自定义策略，使评测脚本无需关心具体类名和构造细节。

## 文档

### `docs/DESIGN.md`

设计说明文档。详细解释问题建模、环境状态、动作定义、合法性约束、调度动力学、策略接口、奖励设计、训练流程、评估指标和复现设置。

### `docs/FILE_OVERVIEW.md`

当前文件。逐一说明每个项目文件的职责，方便评审快速定位代码模块。

## 示例与脚本

### `examples/custom_policy.py`

用户自定义策略示例。实现 `FastestReadyScheduler`，展示如何通过统一接口接入外部调度方法，并可通过 `examples.custom_policy:FastestReadyScheduler` 形式加载评测。

### `scripts/run_linux.sh`

Linux 一键运行脚本。适用于 openEuler、openKylin 以及其他常见 Linux 发行版，内部调用 `python3 run_all.py --config configs/default.json`。

### `scripts/run_windows.ps1`

Windows PowerShell 一键运行脚本，便于当前开发环境直接复现实验。

### `scripts/evaluate_complex.py`

复杂场景泛化验证脚本。使用默认配置训练 RL 模型，再在 `configs/complex.json` 定义的更大 DAG、更密依赖和更多资源场景上评估各策略，并写入 `results/complex/`。

## 测试

### `tests/smoke_test.py`

无需 pytest 的最小自检脚本。会生成快速模式场景、训练一个小模型、评测 HEFT、lookahead HEFT 与 RL，并断言 summary 中包含必要指标且 RL 不差于 HEFT。

## 运行时输出目录

### `artifacts/`

实验中间产物目录。保存生成的场景、训练日志和模型文件，其中 `artifacts/scenarios/` 放数据集，`artifacts/models/` 放策略参数。

### `results/`

评测结果目录。`summary.json` 保存标准化汇总指标，`details.csv` 保存每个验证场景的逐策略结果。
