# 设计说明

## 问题建模

输入场景由三部分组成：任务 DAG、异构计算资源、资源间通信带宽。任务节点包含计算量 `work`，有向边包含跨任务数据量 `data`。资源节点包含类型 `cloud`、`edge`、`device` 和计算速度 `speed`。如果父任务和子任务分配到不同资源，子任务最早开始时间需要额外考虑 `data / bandwidth[src][dst]`。

## 环境状态

`SchedulingEnv` 保存已调度任务、每个资源的可用时间、每个任务的开始/结束时间和当前 makespan。状态没有被固定成一个全局向量，而是通过 `features.py` 为每个合法 `(task, resource)` 动作提取 pairwise 特征，便于适配不同任务数和资源数。

## 动作定义与合法性

动作是 `(task_id, resource_id)`。其中 `task_id` 必须属于 ready task，即所有父任务均已调度；`resource_id` 必须属于场景资源集合。`env.legal_actions()` 返回所有合法组合，`env.legal_action_mask()` 返回 ready/task mask。训练和推理阶段的 RL softmax 都只在合法动作上计算；如果外部策略传入非法动作，`env.step()` 会直接抛出错误，便于尽早发现策略问题。

## 调度动力学

执行动作时，环境按照列表调度规则计算：

```text
start = max(resource_available[resource],
            max(parent_finish + communication_time(parent_resource, resource)))
finish = start + task.work / resource.speed
```

所有任务调度完成后，`max(finish)` 即 makespan。

## 策略接口

统一接口定义在 `cloud_edge_drl/policies/base.py`：

```python
class Scheduler:
    name = "scheduler"
    def reset(self, scenario): ...
    def choose_action(self, env): ...
```

内置策略：

- `HEFTScheduler`：外部启发式基线，按 upward rank 选 ready task，再选 earliest finish resource。
- `LookaheadHEFTScheduler`：枚举当前合法动作，执行一步后用 HEFT 补全剩余调度，选择最终 makespan 最小的动作。
- `PortfolioScheduler`：离线运行 HEFT、lookahead HEFT、min-min、max-min，选择完整 makespan 最低的计划执行。
- `MaxMinScheduler` / `MinMinScheduler`：经典异构调度启发式基线。
- `RandomScheduler`：随机合法动作基线。
- `MLPActorScheduler`：两层 MLP Actor，使用 action masking；默认推理模式为 `rollout_safe`，即 MLP 先给合法动作打分，再对高分候选和 rollout 教师动作做 HEFT 补全校验，选择补全 makespan 最低的动作。

评测还支持 `module:ClassName` 加载用户自定义策略。

## 奖励与训练

环境单步奖励为 makespan 增量的负值，便于调试。训练脚本使用 episode 级目标：

```text
return = - final_makespan / lower_bound
advantage = return - moving_baseline
```

训练流程包括两阶段：

1. 行为克隆预热：默认用 `lookahead_heft` 产生专家动作，更新 MLP 使其优先选择强启发式动作。
2. Masked REINFORCE：在训练场景上采样合法动作，按 episode advantage 更新策略。
3. Rollout-safe 推理：评估时不直接采用 MLP 贪心动作，而是用 learned score 形成候选集合，并保留 `lookahead_heft` 安全候选，再以 HEFT 补全后的最终 makespan 作为推理选择依据。

这样既能保证初始策略不完全随机，也保留了 RL 对场景分布继续优化的空间，同时避免轻量 MLP 在未见复杂场景上出现明显差于 HEFT 的贪心决策。

## 评估指标

验证阶段固定场景集，所有策略共享同一批场景。核心指标：

```text
mean_ratio = mean(policy_makespan / HEFT_makespan)
```

同时输出 `mean_makespan`、`std_makespan`、`std_ratio` 和样本数。逐场景明细写入 `results/details.csv`，汇总写入 `results/summary.json`。

## 复现设置

随机种子、任务规模、资源数量、速度范围、带宽矩阵和训练超参均在 `configs/default.json` 中配置。默认配置会分别生成训练集与验证集，验证集使用不同随机种子偏移，避免训练和评测场景重合。

复杂泛化验证使用 `configs/complex.json` 与 `scripts/evaluate_complex.py`。复杂配置提高任务数量、依赖密度、资源数量和跨节点通信压力，用于检查策略是否只对默认小规模场景过拟合。
