"""masked MLP Actor 的训练流程。

训练分两步：
1. 使用 HEFT 行为克隆预热，让策略先学到一个可用的调度顺序；
2. 使用 masked REINFORCE 在训练场景上继续优化 makespan。

这种设计兼顾工程稳定性和强化学习可训练性，适合比赛框架演示和进一步扩展。
"""

import os
import random
from typing import Any, Dict, List, Tuple

from .env import SchedulingEnv
from .policies.heft import HEFTScheduler
from .policies.lookahead_heft import LookaheadHEFTScheduler
from .policies.rl_policy import MLPActorScheduler
from .utils import clamp, mean, write_json


def _run_policy_episode(scenario, scheduler) -> Tuple[float, List[Dict[str, object]]]:
    """运行一个完整 episode，并记录策略动作。

    当前主流程没有调用它，保留该函数便于后续做离线分析或调试不同策略轨迹。
    """
    env = SchedulingEnv(scenario)
    scheduler.reset(scenario)
    decisions = []
    while not env.done:
        action = scheduler.choose_action(env)
        decisions.append({"action": action})
        env.step(action)
    return env.makespan, decisions


def _build_teacher(name):
    normalized = str(name or "heft").strip().lower()
    if normalized == "heft":
        return HEFTScheduler()
    if normalized in ("lookahead_heft", "lookahead-heft", "rollout_heft", "rollout-heft"):
        return LookaheadHEFTScheduler()
    raise ValueError("Unknown behavior cloning teacher: %s" % name)


def _behavior_clone(policy, scenarios, epochs, learning_rate, teacher_name="heft") -> Dict[str, float]:
    """用 HEFT 作为专家进行行为克隆预训练。

    每一步让学生策略在同一合法动作集合中提高 HEFT 所选动作的概率。
    这样 RL 初始策略不会完全随机，训练方差更小。
    """
    teacher = _build_teacher(teacher_name)
    updates = 0
    exact_matches = 0
    total = 0
    for _epoch in range(int(epochs)):
        for scenario in scenarios:
            env = SchedulingEnv(scenario)
            teacher.reset(scenario)
            while not env.done:
                # teacher action 一定来自同一个环境状态下的合法动作集合。
                action = teacher.choose_action(env)
                actions = env.legal_actions()
                features = policy_feature_batch(env)
                chosen_index = actions.index(action)
                # 统计行为克隆前的贪心预测命中率，作为训练日志参考。
                predicted = policy.choose_action(env)
                if predicted == action:
                    exact_matches += 1
                policy.update_choice(features, chosen_index, advantage=1.0, learning_rate=learning_rate)
                env.step(action)
                updates += 1
                total += 1
    return {
        "bc_teacher": teacher.name,
        "bc_updates": updates,
        "bc_match_rate": exact_matches / float(max(1, total)),
    }


def policy_feature_batch(env):
    """延迟导入特征函数，减少模块初始化时的循环依赖风险。"""
    from .features import legal_action_features

    return legal_action_features(env)


def train_rl(config: Dict[str, Any], train_scenarios, model_path: str, log_path: str) -> MLPActorScheduler:
    """训练 RL 策略并保存模型与训练日志。"""
    seed = int(config["seed"])
    rng = random.Random(seed + 17)
    training = config["training"]
    policy = MLPActorScheduler(
        hidden_size=int(training["hidden_size"]),
        seed=seed + 23,
        inference_mode=training.get("inference_mode", "rollout_safe"),
        rollout_top_k=int(training.get("rollout_top_k", 4)),
    )

    # 第一阶段：模仿 HEFT，获得合理的初始策略。
    bc_stats = _behavior_clone(
        policy,
        train_scenarios,
        epochs=int(training["bc_epochs"]),
        learning_rate=float(training["bc_learning_rate"]),
        teacher_name=training.get("teacher_policy", "heft"),
    )
    base, ext = os.path.splitext(model_path)
    bc_model_path = "%s_bc_only%s" % (base, ext or ".json")
    policy.save(
        bc_model_path,
        metadata={
            "seed": seed,
            "train_scenario_count": len(train_scenarios),
            "algorithm": "behavior cloning only",
            "bc_teacher": bc_stats["bc_teacher"],
        },
    )

    log = {
        "seed": seed,
        "episodes": [],
        "behavior_cloning": bc_stats,
        "hyperparameters": training,
    }

    baseline = None
    momentum = float(training["baseline_momentum"])
    learning_rate = float(training["learning_rate"])
    advantage_clip = float(training["advantage_clip"])

    for episode in range(int(training["episodes"])):
        # 每个 episode 从训练场景集中随机抽一个场景，提高对任务规模变化的泛化能力。
        scenario = rng.choice(train_scenarios)
        env = SchedulingEnv(scenario)
        trajectory = []
        while not env.done:
            action, decision = policy.sample_action(env)
            trajectory.append(decision)
            env.step(action)

        # 使用 makespan 下界做归一化，缓解不同规模场景回报量级不一致的问题。
        normalized_return = -env.makespan / env.lower_bound()
        if baseline is None:
            baseline = normalized_return
        # moving baseline 降低 REINFORCE 方差，advantage 裁剪避免单个坏样本更新过猛。
        advantage = clamp(normalized_return - baseline, advantage_clip)
        baseline = momentum * baseline + (1.0 - momentum) * normalized_return

        # episode 结束后，用同一个 advantage 更新整条轨迹上的动作选择。
        for decision in trajectory:
            policy.update_choice(
                decision["features"],
                decision["chosen_index"],
                advantage=advantage,
                learning_rate=learning_rate,
            )

        log["episodes"].append(
            {
                "episode": episode + 1,
                "scenario_id": scenario.scenario_id,
                "makespan": env.makespan,
                "lower_bound": env.lower_bound(),
                "normalized_return": normalized_return,
                "advantage": advantage,
                "baseline": baseline,
            }
        )

    metadata = {
        "seed": seed,
        "train_scenario_count": len(train_scenarios),
        "mean_train_episode_makespan": mean(item["makespan"] for item in log["episodes"]),
        "algorithm": "HEFT behavior cloning + masked REINFORCE",
    }
    # 模型和日志分开保存：模型供评测加载，日志供报告训练过程。
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    policy.save(model_path, metadata=metadata)
    write_json(log_path, log)
    return policy
