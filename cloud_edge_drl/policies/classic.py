"""Classic heterogeneous scheduling heuristics used as baselines."""

from .base import Scheduler


class OLBScheduler(Scheduler):
    """Opportunistic Load Balancing: assign the next ready task to the earliest free resource."""

    name = "olb"

    def reset(self, scenario):
        pass

    def choose_action(self, env):
        best_key = None
        best_action = None
        for action in env.legal_actions():
            resource_ready = env.resource_available[action[1]]
            estimate = env.estimate(*action)
            key = (resource_ready, estimate["finish"], action[0], action[1])
            if best_key is None or key < best_key:
                best_key = key
                best_action = action
        return best_action


class METScheduler(Scheduler):
    """Minimum Execution Time: choose the task-resource pair with the shortest compute time."""

    name = "met"

    def reset(self, scenario):
        pass

    def choose_action(self, env):
        best_key = None
        best_action = None
        for action in env.legal_actions():
            task = env.scenario.task_by_id[action[0]]
            resource = env.scenario.resource_by_id[action[1]]
            duration = task.work / resource.speed
            estimate = env.estimate(*action)
            key = (duration, estimate["finish"], estimate["start"], action[0], action[1])
            if best_key is None or key < best_key:
                best_key = key
                best_action = action
        return best_action


class MCTScheduler(Scheduler):
    """Minimum Completion Time: choose the ready task-resource pair with the earliest finish."""

    name = "mct"

    def reset(self, scenario):
        pass

    def choose_action(self, env):
        best_key = None
        best_action = None
        for action in env.legal_actions():
            estimate = env.estimate(*action)
            key = (estimate["finish"], estimate["start"], action[0], action[1])
            if best_key is None or key < best_key:
                best_key = key
                best_action = action
        return best_action


class MinMinScheduler(Scheduler):
    """Choose the ready task-resource pair with the smallest earliest finish time."""

    name = "min_min"

    def reset(self, scenario):
        pass

    def choose_action(self, env):
        best_key = None
        best_action = None
        for action in env.legal_actions():
            estimate = env.estimate(*action)
            key = (estimate["finish"], estimate["start"], action[0], action[1])
            if best_key is None or key < best_key:
                best_key = key
                best_action = action
        return best_action


class MaxMinScheduler(Scheduler):
    """Choose the task whose best resource still has the largest earliest finish time."""

    name = "max_min"

    def reset(self, scenario):
        pass

    def choose_action(self, env):
        task_best_actions = []
        for task_id in env.ready_tasks():
            best_key = None
            best_action = None
            for resource in env.scenario.resources:
                action = (task_id, resource.resource_id)
                estimate = env.estimate(*action)
                key = (estimate["finish"], estimate["start"], action[1])
                if best_key is None or key < best_key:
                    best_key = key
                    best_action = action
            task_best_actions.append((best_key[0], -task_id, best_action))
        return max(task_best_actions)[2]
