"""策略工厂和自定义调度器加载器。

评测脚本通过字符串指定策略，例如 `heft`、`random`、`rl`，也可以用
`module:ClassName` 接入用户自己的方法。这样可以在同一批验证场景上公平比较不同策略。
"""

import importlib
import importlib.util
import os

from .classic import MCTScheduler, METScheduler, MaxMinScheduler, MinMinScheduler, OLBScheduler
from .heft import HEFTScheduler
from .lookahead_heft import LookaheadHEFTScheduler
from .portfolio import PortfolioScheduler
from .random_policy import RandomScheduler
from .rl_policy import MLPActorScheduler


def _load_custom(spec):
    """加载用户自定义策略。

    支持两种形式：
    1. `package.module:ClassName`
    2. `path/to/file.py:ClassName`
    """
    module_name, class_name = spec.rsplit(":", 1)
    if module_name.endswith(".py") or os.path.exists(module_name):
        path = os.path.abspath(module_name)
        dynamic_name = "custom_scheduler_%d" % abs(hash(path))
        module_spec = importlib.util.spec_from_file_location(dynamic_name, path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    scheduler_cls = getattr(module, class_name)
    return scheduler_cls()


def build_scheduler(spec, model_path=None, seed=0):
    """根据策略字符串创建调度器实例。"""
    normalized = spec.strip()
    lowered = normalized.lower()
    if lowered == "heft":
        return HEFTScheduler()
    if lowered in ("lookahead_heft", "lookahead-heft", "rollout_heft", "rollout-heft"):
        return LookaheadHEFTScheduler()
    if lowered in ("min_min", "min-min", "minmin"):
        return MinMinScheduler()
    if lowered in ("max_min", "max-min", "maxmin"):
        return MaxMinScheduler()
    if lowered == "olb":
        return OLBScheduler()
    if lowered == "met":
        return METScheduler()
    if lowered == "mct":
        return MCTScheduler()
    if lowered == "portfolio":
        return PortfolioScheduler()
    if lowered == "random":
        return RandomScheduler(seed=seed)
    if lowered in ("rl", "rl_rollout_safe", "rl-safe", "rl_safe"):
        if not model_path:
            raise ValueError("RL policy requires a model path.")
        scheduler = MLPActorScheduler.load(model_path)
        scheduler.inference_mode = "rollout_safe"
        scheduler.name = "rl" if lowered == "rl" else "rl_rollout_safe"
        return scheduler
    if lowered in ("rl_greedy", "rl-greedy", "rl_no_rollout", "rl-no-rollout"):
        if not model_path:
            raise ValueError("RL greedy policy requires a model path.")
        scheduler = MLPActorScheduler.load(model_path)
        scheduler.inference_mode = "greedy"
        scheduler.name = "rl_greedy"
        return scheduler
    if lowered in ("rl_bc_only", "rl-bc-only", "bc_only", "bc-only"):
        if not model_path:
            raise ValueError("RL BC-only policy requires a model path.")
        base, ext = os.path.splitext(model_path)
        bc_model_path = "%s_bc_only%s" % (base, ext or ".json")
        if not os.path.exists(bc_model_path):
            raise ValueError("BC-only model not found: %s. Run training again first." % bc_model_path)
        scheduler = MLPActorScheduler.load(bc_model_path)
        scheduler.inference_mode = "greedy"
        scheduler.name = "rl_bc_only"
        return scheduler
    if ":" in normalized:
        return _load_custom(normalized)
    raise ValueError("Unknown policy spec: %s" % spec)
