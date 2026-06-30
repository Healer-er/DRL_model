"""动作特征提取。

不同场景的任务数和资源数会变化，直接把整个状态拼成固定向量并不方便。本框架采用
pairwise action feature：对每个合法 `(task, resource)` 动作提取同样长度的特征，
RL 策略对这些候选动作分别打分。这也是许多调度学习方法常用的建模方式。
"""

from functools import lru_cache
from typing import Dict, List

from .domain import Scenario
from .env import SchedulingEnv

FEATURE_DIM = 17


@lru_cache(maxsize=256)
def upward_ranks_from_id(scenario_identity: int, scenario: Scenario) -> Dict[int, float]:
    """带缓存的 upward rank 计算入口。

    `scenario_identity` 只用于让 lru_cache 区分不同 Scenario 实例。
    """
    return compute_upward_ranks(scenario)


def compute_upward_ranks(scenario: Scenario) -> Dict[int, float]:
    """计算 HEFT 中使用的 upward rank。

    upward rank 越大，表示该任务到出口任务的平均剩余代价越高，通常应被更早考虑。
    """
    avg_speed = scenario.avg_speed()
    avg_bandwidth = scenario.avg_bandwidth()
    ranks = {}
    # 任务 id 已按拓扑序生成，反向遍历即可保证子节点 rank 已经算好。
    for task in reversed(scenario.tasks):
        children = scenario.children[task.task_id]
        if not children:
            child_rank = 0.0
        else:
            child_rank = max(
                scenario.edge_data[(task.task_id, child)] / avg_bandwidth + ranks[child]
                for child in children
            )
        ranks[task.task_id] = task.work / avg_speed + child_rank
    return ranks


def get_upward_ranks(scenario: Scenario) -> Dict[int, float]:
    """获取场景的 upward rank 缓存结果。"""
    return upward_ranks_from_id(id(scenario), scenario)


def pair_features(env: SchedulingEnv, task_id: int, resource_id: str, ranks: Dict[int, float] = None) -> List[float]:
    """为一个候选动作构造固定长度特征向量。

    特征尽量覆盖三类信息：任务结构重要性、资源异构性、当前调度状态。所有连续值都做
    简单归一化，便于纯 Python MLP 在没有外部深度学习框架时也能稳定训练。
    """
    scenario = env.scenario
    task = scenario.task_by_id[task_id]
    resource = scenario.resource_by_id[resource_id]
    ranks = ranks or get_upward_ranks(scenario)
    estimate = env.estimate(task_id, resource_id)

    task_count = float(max(1, scenario.task_count))
    max_work = max(1.0, scenario.max_work())
    max_speed = max(1.0, scenario.max_speed())
    max_rank = max(1.0, max(ranks.values()) if ranks else 1.0)
    horizon = max(env.lower_bound(), env.makespan, 1.0)
    scheduled_on_resource = sum(1 for value in env.assignment.values() if value == resource_id)

    kind_cloud = 1.0 if resource.kind == "cloud" else 0.0
    kind_edge = 1.0 if resource.kind == "edge" else 0.0
    kind_device = 1.0 if resource.kind == "device" else 0.0

    parents = scenario.parents[task_id]
    inbound_data = 0.0
    for parent in parents:
        inbound_data += scenario.edge_data.get((parent, task_id), 0.0)
    avg_inbound_data = inbound_data / float(max(1, len(parents)))

    return [
        # 任务本身的规模和 DAG 结构特征。
        task.work / max_work,
        float(len(parents)) / task_count,
        float(len(scenario.children[task_id])) / task_count,
        ranks[task_id] / max_rank,
        avg_inbound_data / max(1.0, max(edge.data for edge in scenario.edges) if scenario.edges else 1.0),
        # 当前 episode 进度。
        float(scenario.task_count - len(env.scheduled)) / task_count,
        # 资源计算能力和当前排队状态。
        resource.speed / max_speed,
        env.resource_available[resource_id] / horizon,
        # 如果采取该动作，预计产生的时间影响。
        estimate["start"] / horizon,
        estimate["finish"] / horizon,
        estimate["duration"] / horizon,
        estimate["communication_wait"] / horizon,
        float(scheduled_on_resource) / task_count,
        # 资源层级 one-hot，帮助策略区分云/边/端。
        kind_cloud,
        kind_edge,
        kind_device,
        # 常数 bias 特征，等价于给线性层提供截距输入。
        1.0,
    ]


def legal_action_features(env: SchedulingEnv) -> List[List[float]]:
    """按 `env.legal_actions()` 的顺序返回每个合法动作的特征。"""
    ranks = get_upward_ranks(env.scenario)
    return [pair_features(env, task_id, resource_id, ranks) for task_id, resource_id in env.legal_actions()]
