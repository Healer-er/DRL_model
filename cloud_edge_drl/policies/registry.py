"""策略工厂和自定义调度器加载器。

评测脚本通过字符串指定策略，例如 `heft`、`random`、`rl`，也可以用
`module:ClassName` 接入用户自己的方法。这样可以在同一批验证场景上公平比较不同策略。
"""

import importlib
import importlib.util
import os

from .classic import MaxMinScheduler, MinMinScheduler
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
    if lowered == "portfolio":
        return PortfolioScheduler()
    if lowered == "random":
        return RandomScheduler(seed=seed)
    if lowered == "rl":
        if not model_path:
            raise ValueError("RL policy requires a model path.")
        return MLPActorScheduler.load(model_path)
    if ":" in normalized:
        return _load_custom(normalized)
    raise ValueError("Unknown policy spec: %s" % spec)
