"""用户自定义策略示例。

运行 `python -m cloud_edge_drl.cli evaluate --policies heft,examples.custom_policy:FastestReadyScheduler`
即可把该策略接入统一评测流程。
"""

from cloud_edge_drl.policies.base import Scheduler


class FastestReadyScheduler(Scheduler):
    """选择当前计算量最大的 ready task，并放到最快资源上。"""

    name = "fastest_ready"

    def choose_action(self, env):
        """实现统一策略接口，返回一个合法调度动作。"""
        ready = env.ready_tasks()
        task_id = max(ready, key=lambda task: env.scenario.task_by_id[task].work)
        fastest = max(env.scenario.resources, key=lambda resource: resource.speed)
        best_action = (task_id, fastest.resource_id)
        if best_action in env.legal_actions():
            return best_action
        return env.legal_actions()[0]
