# 竞赛优化与测试计划

本文档面向“基于深度强化学习的云-边-端异构计算资源管理调度方法”赛题，记录当前项目为提升评审说服力新增的优化点、推荐评测命令和后续测试场景。

## 已完成的工程增强

1. RL 消融策略

- `rl_bc_only`：只使用行为克隆阶段保存的模型，评测时直接贪心选动作。
- `rl_greedy`：使用完整训练后的 MLP，评测时不做 rollout-safe 校验。
- `rl`：保持原有 rollout-safe 推理模式，用 MLP 生成候选并用 HEFT rollout 做安全选择。

这三组结果可以说明“行为克隆、强化学习更新、安全 rollout”分别带来的影响，避免评委认为 RL 结果只是启发式策略的别名。

2. 更多经典基线

新增异构调度常用基线：

- `olb`：Opportunistic Load Balancing。
- `met`：Minimum Execution Time。
- `mct`：Minimum Completion Time。

默认评测策略已扩展为：

```text
heft, lookahead_heft, portfolio, min_min, max_min, mct, met, olb, random, rl_bc_only, rl_greedy, rl
```

3. 多 seed 泛化评测脚本

新增脚本：

```bash
python scripts/benchmark_generalization.py --config configs/default.json --seeds 2026,2027,2028
```

输出：

- `results/generalization/aggregate.json`
- `results/generalization/report.md`

报告会汇总每个策略在多 seed、多验证场景上的 `mean_ratio`、`std_ratio`、最好/最差 ratio、优于 HEFT 的比例和不差于 HEFT 的比例。

4. 基础正确性测试

新增：

```bash
python tests/test_env_correctness.py
```

覆盖：

- ready task 依赖约束；
- 非法动作拒绝；
- 跨资源通信等待；
- 同资源无通信等待；
- makespan 等于所有任务完成时间最大值。

## 推荐正式评测命令

快速自检：

```bash
python tests/smoke_test.py
python tests/test_env_correctness.py
python -m cloud_edge_drl.cli --quick run-all
```

默认规模完整链路：

```bash
python run_all.py --config configs/default.json
```

复杂场景：

```bash
python scripts/evaluate_complex.py
```

多 seed 泛化评测：

```bash
python scripts/benchmark_generalization.py ^
  --config configs/default.json ^
  --seeds 2026,2027,2028,2029,2030 ^
  --output-root results/generalization_default
```

复杂配置多 seed 评测：

```bash
python scripts/benchmark_generalization.py ^
  --config configs/complex.json ^
  --seeds 3026,3027,3028 ^
  --output-root results/generalization_complex
```

## 后续最需要补充的测试场景

1. DAG 结构泛化

- 完全并行 DAG；
- 完全链式 DAG；
- fork-join DAG；
- 星型依赖；
- 高依赖密度 DAG；
- 多入口、多出口 DAG；
- 小规模 DAG 的最优解对比。

2. 资源异构压力

- 云端极快但云-端带宽很低；
- 边缘资源数量多但速度中等；
- 终端资源慢但本地通信快；
- 资源速度差异极小；
- 资源速度差异极大；
- 单资源、单云多边、多云多边多端。

3. 通信压力

- compute-heavy：计算量大、通信量小；
- communication-heavy：通信量大、计算量小；
- 非对称带宽矩阵；
- cloud-device 瓶颈；
- edge-device 瓶颈；
- 高数据量关键路径。

4. 训练与推理消融

- `rl_bc_only` vs `rl_greedy` vs `rl`；
- 不同 `rollout_top_k`；
- 不同 `hidden_size`；
- 不同训练 episode 数；
- 不同行为克隆 teacher；
- 小图训练、大图测试；
- 稀疏图训练、稠密图测试。

5. 工程稳定性

- 模型文件缺失；
- 配置字段缺失或类型错误；
- 自定义策略加载失败；
- 空 DAG、单任务 DAG；
- 不同 Python 版本；
- openEuler / openKylin / Ubuntu 兼容运行。

## 竞赛材料建议

正式提交时建议把 `report.md` 中的表格转为论文或答辩材料中的“多 seed 泛化实验表”，并补充甘特图或箱线图。重点解释：

- `rl_bc_only` 和 `rl_greedy` 是学习模型自身能力的消融；
- `rl` 是面向工程稳定性的安全推理模式；
- 所有策略共享同一个 `SchedulingEnv`，因此 makespan、通信时间和合法动作约束完全一致；
- 多 seed、多规模、多通信压力实验用于证明泛化能力，而不是只在固定 10 个验证场景上调参。
