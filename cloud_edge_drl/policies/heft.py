"""HEFT 风格外部基线策略。

HEFT 是异构计算任务调度中的经典启发式方法。这里实现一个列表调度版本：
先用 upward rank 衡量任务优先级，再为该任务选择最早完成的资源。
"""

from .base import Scheduler
from ..features import compute_upward_ranks


class HEFTScheduler(Scheduler):
    """选择 highest upward-rank ready task，再选择 earliest-finish resource。"""

    name = "heft"

    def __init__(self):
        """ranks 会在每个场景 reset 时按当前 DAG 重新计算。"""
        self.ranks = None

    def reset(self, scenario):
        """为当前场景预计算 upward rank。"""
        self.ranks = compute_upward_ranks(scenario)

    def choose_action(self, env):
        """返回 HEFT 对当前状态的下一步调度动作。"""
        if self.ranks is None:
            self.reset(env.scenario)
        ready = env.ready_tasks()
        # ready task 中 rank 越大，说明它后续关键路径越重，优先调度。
        task_id = max(ready, key=lambda task: (self.ranks[task], -task))
        best_resource = None
        best_finish = None
        best_start = None
        for resource in env.scenario.resources:
            estimate = env.estimate(task_id, resource.resource_id)
            # 先比较完成时间，再比较开始时间，最后偏向速度更快的资源，保证确定性。
            key = (estimate["finish"], estimate["start"], -resource.speed)
            if best_finish is None or key < (best_finish, best_start, -env.scenario.resource_by_id[best_resource].speed):
                best_resource = resource.resource_id
                best_finish = estimate["finish"]
                best_start = estimate["start"]
        return task_id, best_resource
