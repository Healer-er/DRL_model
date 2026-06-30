"""带合法动作约束的列表调度环境。

环境负责仿真“选择一个 ready task，并把它放到一个资源上”的过程。它不关心策略来自
HEFT、随机还是 RL，只检查动作是否合法并更新资源时间线。这种拆分保证训练和评测阶段
使用完全相同的调度规则。
"""

from typing import Dict, List, Optional, Tuple

from .domain import Scenario

Action = Tuple[int, str]


class SchedulingEnv:
    """云-边-端异构资源上的 DAG 列表调度器。"""

    def __init__(self, scenario: Optional[Scenario] = None) -> None:
        self.scenario = None
        if scenario is not None:
            self.reset(scenario)

    def reset(self, scenario: Scenario) -> "SchedulingEnv":
        """载入一个新场景并清空所有调度状态。"""
        self.scenario = scenario
        # scheduled 记录已经完成“资源分配决策”的任务，不代表真实时间已经推进到完成点。
        self.scheduled = set()
        # assignment/task times 是评测和后续子任务通信时间计算的依据。
        self.assignment = {}
        self.start_times = {}
        self.finish_times = {}
        # 每个资源当前可用时间，决定新任务是否需要排队等待。
        self.resource_available = {resource.resource_id: 0.0 for resource in scenario.resources}
        self.timeline = []
        self.makespan = 0.0
        return self

    @property
    def done(self) -> bool:
        """所有任务都已经被调度时，一个 episode 结束。"""
        return len(self.scheduled) == self.scenario.task_count

    def ready_tasks(self) -> List[int]:
        """返回当前所有满足依赖约束的任务。

        ready 的定义是：该任务尚未调度，且所有父任务都已经完成调度决策。
        """
        ready = []
        for task in self.scenario.tasks:
            if task.task_id in self.scheduled:
                continue
            if all(parent in self.scheduled for parent in self.scenario.parents[task.task_id]):
                ready.append(task.task_id)
        return ready

    def legal_actions(self) -> List[Action]:
        """枚举所有合法动作 `(ready_task, resource)`。

        这是 action masking 的来源。RL 策略只会对这里返回的动作打分和归一化。
        """
        actions = []
        for task_id in self.ready_tasks():
            for resource in self.scenario.resources:
                actions.append((task_id, resource.resource_id))
        return actions

    def legal_action_mask(self) -> List[List[bool]]:
        """返回二维合法动作 mask。

        第一维对应任务，第二维对应资源。由于资源只要存在就可用，所以 mask 的关键在于
        task 是否 ready；该接口便于替换成更标准的 Gym 风格环境。
        """
        ready = set(self.ready_tasks())
        return [
            [task.task_id in ready for _resource in self.scenario.resources]
            for task in self.scenario.tasks
        ]

    def estimate(self, task_id: int, resource_id: str) -> Dict[str, float]:
        """估计把某个 ready task 放到某个资源上的开始/结束时间。

        该函数只做预测，不改变环境状态，因此 HEFT、RL 特征提取和调试代码都可以安全调用。
        """
        if task_id not in self.scenario.task_by_id:
            raise ValueError("Unknown task id %s." % task_id)
        if resource_id not in self.scenario.resource_by_id:
            raise ValueError("Unknown resource id %s." % resource_id)

        task = self.scenario.task_by_id[task_id]
        resource = self.scenario.resource_by_id[resource_id]
        dependency_ready_time = 0.0
        communication_wait = 0.0
        for parent in self.scenario.parents[task_id]:
            parent_resource = self.assignment[parent]
            communication = self.scenario.communication_time(parent, task_id, parent_resource, resource_id)
            communication_wait = max(communication_wait, communication)
            # 子任务必须等待所有父任务完成，并等待跨资源数据传输结束。
            dependency_ready_time = max(dependency_ready_time, self.finish_times[parent] + communication)
        # 任务开始时间同时受资源排队时间和依赖完成时间约束。
        start = max(self.resource_available[resource_id], dependency_ready_time)
        duration = task.work / resource.speed
        finish = start + duration
        return {
            "start": start,
            "finish": finish,
            "duration": duration,
            "communication_wait": communication_wait,
        }

    def step(self, action: Action) -> Tuple[Dict[str, float], float, bool, Dict[str, object]]:
        """执行一个合法调度动作并更新环境。

        返回形式模仿强化学习环境：`observation, reward, done, info`。
        """
        if self.done:
            raise ValueError("Cannot step a finished environment.")
        task_id, resource_id = action
        if action not in self.legal_actions():
            # 外部策略如果绕过 mask 选了非法动作，直接报错，避免 silently 产生不公平结果。
            raise ValueError("Illegal action %r. Use env.legal_actions() or env.legal_action_mask()." % (action,))

        before = self.makespan
        estimate = self.estimate(task_id, resource_id)
        # 真正提交调度决策：记录分配、时间线，并推进资源可用时间。
        self.scheduled.add(task_id)
        self.assignment[task_id] = resource_id
        self.start_times[task_id] = estimate["start"]
        self.finish_times[task_id] = estimate["finish"]
        self.resource_available[resource_id] = estimate["finish"]
        self.makespan = max(self.makespan, estimate["finish"])
        self.timeline.append(
            {
                "task": task_id,
                "resource": resource_id,
                "start": estimate["start"],
                "finish": estimate["finish"],
            }
        )
        # 单步奖励是 makespan 增量的负值；训练主流程使用 episode 级归一化 return。
        reward = before - self.makespan
        return self.observation(), reward, self.done, {"estimate": estimate}

    def observation(self) -> Dict[str, float]:
        """给调试或外部环境包装器使用的简洁观测。"""
        return {
            "scheduled_count": len(self.scheduled),
            "remaining_count": self.scenario.task_count - len(self.scheduled),
            "makespan": self.makespan,
        }

    def lower_bound(self) -> float:
        """计算一个粗略 makespan 下界，用于训练回报归一化。

        下界由资源容量约束和关键路径约束共同给出，不要求严格最紧，只要求稳定可解释。
        """
        total_min_compute = sum(task.work / self.scenario.max_speed() for task in self.scenario.tasks)
        resource_bound = total_min_compute / float(self.scenario.resource_count)
        critical = self._critical_path_lower_bound()
        return max(resource_bound, critical, 1.0)

    def _critical_path_lower_bound(self) -> float:
        """按最快计算资源和平均带宽估计关键路径长度。"""
        min_compute = {task.task_id: task.work / self.scenario.max_speed() for task in self.scenario.tasks}
        best = {}
        for task in self.scenario.tasks:
            parent_best = 0.0
            for parent in self.scenario.parents[task.task_id]:
                data = self.scenario.edge_data[(parent, task.task_id)]
                comm = data / self.scenario.avg_bandwidth()
                parent_best = max(parent_best, best[parent] + comm)
            best[task.task_id] = parent_best + min_compute[task.task_id]
        return max(best.values()) if best else 1.0
