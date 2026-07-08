"""纯 Python 实现的 masked MLP Actor 策略。

为了让项目在没有 PyTorch 的系统上也能一键运行，本文件手写了一个很小的两层 MLP。
它对每个合法 `(task, resource)` 动作分别打分，再在合法动作集合上做 softmax。
因此训练和推理阶段都天然满足 ready task / legal action 约束。
"""

import json
import math
import os
import random
from typing import Dict, List, Tuple

from .base import Scheduler
from .heft import HEFTScheduler
from .lookahead_heft import LookaheadHEFTScheduler
from ..features import FEATURE_DIM, legal_action_features
from ..env import SchedulingEnv


class MLPActorScheduler(Scheduler):
    """对合法 task-resource 动作打分的两层 MLP Actor。"""

    name = "rl"

    def __init__(
        self,
        feature_dim=FEATURE_DIM,
        hidden_size=32,
        seed=0,
        inference_mode="rollout_safe",
        rollout_top_k=4,
    ):
        """初始化网络参数。

        网络结构为：`features -> tanh hidden -> scalar score`。score 不是最终概率，
        它会和同一状态下其他合法动作的 score 一起做 softmax。
        """
        self.feature_dim = int(feature_dim)
        self.hidden_size = int(hidden_size)
        self.inference_mode = str(inference_mode)
        self.rollout_top_k = int(rollout_top_k)
        self.rng = random.Random(seed)
        scale = 0.08
        self.w1 = [
            [self.rng.uniform(-scale, scale) for _ in range(self.feature_dim)]
            for _ in range(self.hidden_size)
        ]
        self.b1 = [0.0 for _ in range(self.hidden_size)]
        self.w2 = [self.rng.uniform(-scale, scale) for _ in range(self.hidden_size)]
        self.b2 = 0.0

    def reset(self, scenario):
        """RL 策略当前没有场景级缓存，因此 reset 留空。"""
        pass

    def _forward_one(self, features: List[float]) -> Tuple[float, List[float]]:
        """前向计算单个动作的 score，并返回 hidden 激活供反向更新使用。"""
        hidden = []
        for row, bias in zip(self.w1, self.b1):
            value = bias
            for weight, feature in zip(row, features):
                value += weight * feature
            hidden.append(math.tanh(value))
        score = self.b2
        for weight, activation in zip(self.w2, hidden):
            score += weight * activation
        return score, hidden

    def scores(self, feature_batch: List[List[float]]) -> Tuple[List[float], List[List[float]]]:
        """批量计算一个状态下所有合法动作的 score。"""
        scores = []
        hidden_batch = []
        for features in feature_batch:
            score, hidden = self._forward_one(features)
            scores.append(score)
            hidden_batch.append(hidden)
        return scores, hidden_batch

    @staticmethod
    def softmax(scores: List[float]) -> List[float]:
        """数值稳定的 softmax。

        输入只包含合法动作的 score，所以输出概率也只分布在合法动作上。
        """
        max_score = max(scores)
        exps = [math.exp(max(-40.0, min(40.0, score - max_score))) for score in scores]
        total = sum(exps)
        return [value / total for value in exps]

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

    def _rollout_safe_action(self, env, actions, scores):
        indexed = sorted(range(len(actions)), key=lambda index: scores[index], reverse=True)
        if self.rollout_top_k > 0:
            indexed = indexed[: self.rollout_top_k]

        candidates = {actions[index] for index in indexed}
        # Include a strong constraint-aware teacher action as a safety candidate. The
        # learned actor still proposes candidates, while this guard prevents poor
        # out-of-distribution greedy choices from dominating validation performance.
        teacher = LookaheadHEFTScheduler()
        teacher.reset(env.scenario)
        candidates.add(teacher.choose_action(env))

        best_key = None
        best_action = None
        for action in candidates:
            candidate_env = self._clone_env(env)
            _observation, _reward, _done, info = candidate_env.step(action)
            final_makespan = self._finish_with_heft(candidate_env)
            estimate = info["estimate"]
            actor_rank = -scores[actions.index(action)] if action in actions else 0.0
            key = (
                final_makespan,
                estimate["finish"],
                estimate["start"],
                actor_rank,
                action[0],
                action[1],
            )
            if best_key is None or key < best_key:
                best_key = key
                best_action = action
        return best_action

    def choose_action(self, env):
        """推理阶段：选择 RL 打分动作，并可用 rollout 做安全校验。"""
        actions = env.legal_actions()
        feature_batch = legal_action_features(env)
        scores, _hidden = self.scores(feature_batch)
        if self.inference_mode in ("rollout_safe", "safe_rollout", "hybrid"):
            return self._rollout_safe_action(env, actions, scores)
        best_index = max(range(len(actions)), key=lambda index: scores[index])
        return actions[best_index]

    def sample_action(self, env):
        """训练阶段：按 masked softmax 概率采样一个合法动作。

        返回的 decision 会被训练循环保存，用于 episode 结束后的策略梯度更新。
        """
        actions = env.legal_actions()
        feature_batch = legal_action_features(env)
        scores, _hidden = self.scores(feature_batch)
        probs = self.softmax(scores)
        sample = self.rng.random()
        cumulative = 0.0
        chosen_index = len(actions) - 1
        for index, prob in enumerate(probs):
            cumulative += prob
            if sample <= cumulative:
                chosen_index = index
                break
        return actions[chosen_index], {
            "features": feature_batch,
            "chosen_index": chosen_index,
            "probs": probs,
        }

    def update_choice(self, feature_batch: List[List[float]], chosen_index: int, advantage: float, learning_rate: float) -> None:
        """对一次动作选择执行策略梯度更新。

        目标是最大化 `advantage * log pi(a|s)`。当 advantage 为正时，提高所选动作概率；
        当 advantage 为负时，降低所选动作概率。
        """
        scores, hidden_batch = self.scores(feature_batch)
        probs = self.softmax(scores)
        # log-softmax 对 score 的梯度：one_hot(chosen) - probs。
        grad_scores = [-prob for prob in probs]
        grad_scores[chosen_index] += 1.0

        # 手写反向传播累积梯度，避免引入外部深度学习框架依赖。
        grad_w1 = [[0.0 for _ in range(self.feature_dim)] for _ in range(self.hidden_size)]
        grad_b1 = [0.0 for _ in range(self.hidden_size)]
        grad_w2 = [0.0 for _ in range(self.hidden_size)]
        grad_b2 = 0.0

        for features, hidden, grad_score in zip(feature_batch, hidden_batch, grad_scores):
            scaled = advantage * grad_score
            grad_b2 += scaled
            for hidden_index in range(self.hidden_size):
                grad_w2[hidden_index] += scaled * hidden[hidden_index]
                # tanh'(z) = 1 - tanh(z)^2。
                dz = scaled * self.w2[hidden_index] * (1.0 - hidden[hidden_index] * hidden[hidden_index])
                grad_b1[hidden_index] += dz
                for feature_index in range(self.feature_dim):
                    grad_w1[hidden_index][feature_index] += dz * features[feature_index]

        # 梯度上升：直接朝增大目标函数的方向更新参数。
        for hidden_index in range(self.hidden_size):
            self.w2[hidden_index] += learning_rate * grad_w2[hidden_index]
            self.b1[hidden_index] += learning_rate * grad_b1[hidden_index]
            for feature_index in range(self.feature_dim):
                self.w1[hidden_index][feature_index] += learning_rate * grad_w1[hidden_index][feature_index]
        self.b2 += learning_rate * grad_b2

    def to_dict(self) -> Dict[str, object]:
        """把模型参数转成可 JSON 保存的字典。"""
        return {
            "name": self.name,
            "feature_dim": self.feature_dim,
            "hidden_size": self.hidden_size,
            "inference_mode": self.inference_mode,
            "rollout_top_k": self.rollout_top_k,
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
        }

    @staticmethod
    def from_dict(data: Dict[str, object]) -> "MLPActorScheduler":
        """从 JSON 字典恢复模型参数。"""
        model = MLPActorScheduler(
            feature_dim=int(data["feature_dim"]),
            hidden_size=int(data["hidden_size"]),
            seed=0,
            inference_mode=data.get("inference_mode", "rollout_safe"),
            rollout_top_k=int(data.get("rollout_top_k", 4)),
        )
        model.w1 = data["w1"]
        model.b1 = data["b1"]
        model.w2 = data["w2"]
        model.b2 = float(data["b2"])
        return model

    def save(self, path: str, metadata: Dict[str, object] = None) -> None:
        """保存模型参数和训练元信息。"""
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        payload = {"model": self.to_dict(), "metadata": metadata or {}}
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    @staticmethod
    def load(path: str) -> "MLPActorScheduler":
        """从磁盘加载 RL 策略。"""
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return MLPActorScheduler.from_dict(payload["model"])
