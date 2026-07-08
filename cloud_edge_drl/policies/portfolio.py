"""Portfolio scheduler that selects the best complete plan from several policies."""

from .base import Scheduler
from .classic import MaxMinScheduler, MinMinScheduler
from .heft import HEFTScheduler
from .lookahead_heft import LookaheadHEFTScheduler
from ..env import SchedulingEnv


class PortfolioScheduler(Scheduler):
    """Run multiple schedulers offline and execute the lowest-makespan plan."""

    name = "portfolio"

    @staticmethod
    def _run_plan(scenario, scheduler):
        env = SchedulingEnv(scenario)
        scheduler.reset(scenario)
        while not env.done:
            env.step(scheduler.choose_action(env))
        return env.makespan, [(item["task"], item["resource"]) for item in env.timeline]

    def reset(self, scenario):
        candidates = [
            HEFTScheduler(),
            LookaheadHEFTScheduler(),
            MaxMinScheduler(),
            MinMinScheduler(),
        ]
        best_key = None
        best_plan = None
        best_source = None
        for scheduler in candidates:
            makespan, plan = self._run_plan(scenario, scheduler)
            key = (makespan, scheduler.name)
            if best_key is None or key < best_key:
                best_key = key
                best_plan = plan
                best_source = scheduler.name
        self.plan = best_plan or []
        self.source_policy = best_source

    def choose_action(self, env):
        legal = set(env.legal_actions())
        for action in self.plan:
            if action in legal:
                return action
        scheduler = LookaheadHEFTScheduler()
        scheduler.reset(env.scenario)
        return scheduler.choose_action(env)
