"""One-step rollout scheduler using HEFT as the completion policy."""

from .base import Scheduler
from .heft import HEFTScheduler
from ..env import SchedulingEnv


class LookaheadHEFTScheduler(Scheduler):
    """Choose the legal action with the best HEFT-completed final makespan."""

    name = "lookahead_heft"

    def reset(self, scenario):
        self.scenario = scenario

    @staticmethod
    def _clone_env(env):
        clone = SchedulingEnv(env.scenario)
        clone.scheduled = set(env.scheduled)
        clone.assignment = dict(env.assignment)
        clone.start_times = dict(env.start_times)
        clone.finish_times = dict(env.finish_times)
        clone.resource_available = dict(env.resource_available)
        clone.timeline = list(env.timeline)
        clone.makespan = env.makespan
        return clone

    @staticmethod
    def _finish_with_heft(env):
        scheduler = HEFTScheduler()
        scheduler.reset(env.scenario)
        while not env.done:
            env.step(scheduler.choose_action(env))
        return env.makespan

    def choose_action(self, env):
        best_key = None
        best_action = None
        for action in env.legal_actions():
            candidate_env = self._clone_env(env)
            _observation, _reward, _done, info = candidate_env.step(action)
            final_makespan = self._finish_with_heft(candidate_env)
            estimate = info["estimate"]
            key = (
                final_makespan,
                estimate["finish"],
                estimate["start"],
                action[0],
                action[1],
            )
            if best_key is None or key < best_key:
                best_key = key
                best_action = action
        return best_action
