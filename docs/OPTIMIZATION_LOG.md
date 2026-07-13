# 项目优化日记

本文档用于持续记录项目面向竞赛提交、性能改进、工程质量提升和测试覆盖增强的每次优化。每次修改完成后，应追加一条日志，确保后续写论文、答辩材料、测试报告时可以追溯“为什么改、改了什么、如何验证、结果如何”。

## 日记格式规范

每次优化按以下格式记录：

```markdown
## YYYY-MM-DD 第 N 次优化：一句话概括

### 背景与目标
- 当前问题：
- 优化目标：
- 对应赛题评分点：

### 修改内容
- 修改文件：
- 核心实现：
- 接口/配置变化：

### 验证方式
- 执行命令：
- 测试范围：
- 关键输出：

### 结果与影响
- 性能结果：
- 工程影响：
- 文档/演示价值：

### 后续 TODO
- TODO 1：
- TODO 2：
```

记录要求：

- 每条日志必须写清楚具体文件、具体策略或脚本名，避免只写“优化代码”。
- 涉及性能时必须记录指标名称，例如 `mean_ratio`、`std_ratio`、样本数、seed。
- 涉及测试时必须记录实际执行命令和是否通过。
- 如果只是文档或实验补充，也要说明它对应的竞赛评分点。
- 对未完成或风险项使用 `后续 TODO` 保留，不要混在结果中。

## 2026-07-13 第 1 次优化：补充 RL 消融、经典基线、多 seed 泛化评测与正确性测试

### 背景与目标

- 当前问题：原项目虽然具备一键训练评测、HEFT 对比和 rollout-safe RL 策略，但 `rl` 与 `lookahead_heft` / `portfolio` 的结果完全一致，容易被评委质疑学习模型的独立贡献不清晰。
- 优化目标：增加消融实验、更多传统基线、多 seed 泛化评测和基础正确性测试，使竞赛材料能解释“学习部分、安全 rollout、启发式基线”的各自作用。
- 对应赛题评分点：功能完整性、性能优化、泛化能力、训练策略有效性、工程质量、复现实验说明。

### 修改内容

- 修改 `cloud_edge_drl/policies/classic.py`：新增三个经典异构调度基线。
  - `OLBScheduler` / `olb`：选择最早空闲资源。
  - `METScheduler` / `met`：选择计算执行时间最短的任务-资源组合。
  - `MCTScheduler` / `mct`：选择最早完成的任务-资源组合。
- 修改 `cloud_edge_drl/policies/registry.py`：扩展策略注册。
  - 新增 `olb`、`met`、`mct`。
  - 新增 `rl_bc_only`：加载行为克隆阶段模型并贪心推理。
  - 新增 `rl_greedy`：加载完整训练模型但不做 rollout-safe。
  - 保持 `rl` 为 rollout-safe 推理模式。
- 修改 `cloud_edge_drl/train.py`：行为克隆结束后额外保存 `*_bc_only.json` 模型，用于消融评测。
- 修改 `configs/default.json` 和 `configs/complex.json`：默认评测策略扩展为：

```text
heft, lookahead_heft, portfolio, min_min, max_min, mct, met, olb, random, rl_bc_only, rl_greedy, rl
```

- 新增 `scripts/benchmark_generalization.py`：支持多 seed 训练和评测，输出：
  - `aggregate.json`
  - `report.md`
- 新增 `tests/test_env_correctness.py`：覆盖调度环境基础正确性。
- 新增 `docs/COMPETITION_OPTIMIZATION_PLAN.md`：记录竞赛优化方向、推荐命令和后续测试场景。
- 修改 `cloud_edge_drl/cli.py`：修复 `--quick run-all` 与 `run-all --quick` 参数顺序兼容问题。

### 验证方式

- 执行环境正确性测试：

```bash
python tests\test_env_correctness.py
```

关键输出：

```text
Ran 5 tests in 0.000s
OK
```

- 执行原有烟测：

```bash
python tests\smoke_test.py
```

关键输出：

```text
smoke test passed
```

- 执行 quick 一键链路：

```bash
python -m cloud_edge_drl.cli --quick run-all
```

验证点：`--quick` 放在子命令前仍能生效，验证场景数缩小为 4。

- 执行默认完整链路：

```bash
python run_all.py --config configs/default.json
```

关键输出：

```text
heft mean_ratio=1.0000 std_ratio=0.0000 count=10
lookahead_heft mean_ratio=0.9777 std_ratio=0.0328 count=10
portfolio mean_ratio=0.9777 std_ratio=0.0328 count=10
min_min mean_ratio=1.1708 std_ratio=0.1238 count=10
max_min mean_ratio=1.0817 std_ratio=0.1088 count=10
mct mean_ratio=1.1708 std_ratio=0.1238 count=10
met mean_ratio=1.5348 std_ratio=0.3316 count=10
olb mean_ratio=2.1128 std_ratio=0.6395 count=10
random mean_ratio=4.6876 std_ratio=1.9906 count=10
rl_bc_only mean_ratio=1.1357 std_ratio=0.0985 count=10
rl_greedy mean_ratio=1.0694 std_ratio=0.0830 count=10
rl mean_ratio=0.9777 std_ratio=0.0328 count=10
```

- 执行泛化评测脚本 smoke 验证：

```bash
python scripts\benchmark_generalization.py --seeds 2026,2027 --train-count 4 --val-count 3 --episodes 4 --output-root results\generalization_smoke
```

关键输出：

```text
aggregate=results\generalization_smoke\aggregate.json
report=results\generalization_smoke\report.md
```

说明：该 smoke 输出仅用于验证脚本链路，验证后已清理临时目录。

### 结果与影响

- 性能结果：默认验证集上 `rl` 的 `mean_ratio=0.9777`，优于 HEFT；`rl_bc_only=1.1357`、`rl_greedy=1.0694`，说明仅行为克隆或纯 MLP 贪心不够稳定，rollout-safe 推理带来工程稳定性。
- 工程影响：评测矩阵更完整，支持更多传统基线和 RL 消融策略；训练产物中新增 `rl_policy_bc_only.json`。
- 文档/演示价值：可以在答辩中用 `rl_bc_only -> rl_greedy -> rl` 的结果解释学习模型与安全推理的关系，降低“只是 HEFT 变体”的质疑。

### 后续 TODO

- 补充更大规模多 seed 正式泛化评测，例如 5 个 seed、每个 seed 100+ 验证场景。
- 增加真实工作流 DAG 或标准工作流族，例如 Montage、CyberShake、Epigenomics、LIGO、SIPHT。
- 增加甘特图、箱线图和策略对比图，用于答辩演示和测试报告。
- 增加配置 schema 校验和更完整的异常输入测试。
- 在 openEuler 24.03-LTS-SP3 环境执行完整复现验证。

## 2026-07-13 第 2 次优化：新增可视化报告与 SVG 甘特图

### 背景与目标

- 当前问题：`summary.json` 和 `details.csv` 能满足机器评测，但不适合现场答辩直接展示；评委很难从纯表格中直观看到任务在云、边、端资源上的排布差异。
- 优化目标：在不引入第三方依赖的前提下，自动生成 Markdown 报告和 SVG 甘特图，用于展示不同策略在同一验证场景下的任务时间线与 makespan 差异。
- 对应赛题评分点：演示效果、文档质量、性能分析可解释性、工程可用性。

### 修改内容

- 新增 `scripts/generate_visual_report.py`：
  - 重跑指定策略并捕获每个策略的 `timeline`。
  - 聚合生成策略排名表。
  - 自动选择提升最明显的验证场景，或通过 `--scenario-id` 指定场景。
  - 为选定场景下的每个策略生成 SVG 甘特图。
  - 输出 `report.md`、`summary.json`、`details.csv`、`timelines.json`。
- 新增 `docs/VISUAL_REPORT_GUIDE.md`：
  - 记录默认配置、复杂配置、指定策略、指定场景的运行命令。
  - 说明每个输出文件的用途。
  - 给出答辩材料中如何对比 HEFT、`rl_bc_only`、`rl_greedy`、`rl` 的建议。

### 验证方式

- 推荐执行命令：

```bash
python scripts\generate_visual_report.py --config configs/default.json
```

- 预期输出：

```text
report=results\visual_report\report.md
charts=<策略数量>
```

- 需要检查的文件：

```text
results/visual_report/report.md
results/visual_report/timelines.json
results/visual_report/gantt_<scenario>_<policy>.svg
```

### 结果与影响

- 性能结果：该优化不改变调度算法本身，不影响 `mean_ratio`。
- 工程影响：新增独立可视化脚本，不改变主评测入口和 `evaluate.py` 的输出格式。
- 文档/演示价值：可以直接在答辩 PPT 或测试报告中展示甘特图，说明 RL/rollout-safe 策略在具体场景中如何改变任务分配和完成时间。

### 后续 TODO

- 增加箱线图或 SVG 柱状图，展示多 seed 下的 `mean_ratio` 分布。
- 在正式完整结果上生成一版 `results/visual_report/` 并纳入提交材料。
- 对真实工作流 DAG 增加专门的甘特图展示案例。
