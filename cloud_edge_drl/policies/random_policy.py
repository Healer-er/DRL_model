"""随机合法动作基线。

该策略不追求性能，主要用于验证环境 mask、评测循环和指标聚合是否正常。
"""

import random

from .base import Scheduler


class RandomScheduler(Scheduler):
    name = "random"

    def __init__(self, seed=0):
        """固定随机种子，保证重复评测时结果可复现。"""
        self.rng = random.Random(seed)

    def choose_action(self, env):
        """只从环境暴露的合法动作中随机选择。"""
        actions = env.legal_actions()
        return self.rng.choice(actions)
