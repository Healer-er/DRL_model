"""配置读取与运行模式辅助函数。

赛题要求“环境与资源参数可配置”，因此所有任务规模、资源数量、异构速度和训练超参
都从 JSON 配置读取。本模块集中处理配置相关的小逻辑，避免训练/评测脚本里出现硬编码。
"""

import json
import os
from copy import deepcopy
from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    """读取 JSON 配置文件。"""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并配置。

    当前主流程没有强依赖它，但保留这个函数是为了方便后续实验从命令行或额外文件
    局部覆盖默认配置，例如只改学习率或资源数量。
    """
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def ensure_output_dirs(config: Dict[str, Any]) -> None:
    """根据配置创建所有输出目录。"""
    for key in ("artifact_dir", "scenario_dir", "model_dir", "result_dir"):
        path = config["paths"][key]
        os.makedirs(path, exist_ok=True)


def apply_quick_mode(config: Dict[str, Any]) -> Dict[str, Any]:
    """生成一个更小的配置副本，用于烟测和课堂演示。

    quick 模式会减少场景数、任务数和训练轮数，目标是快速验证链路是否通畅，
    不用于正式性能报告。
    """
    quick = deepcopy(config)
    quick["scenario"]["train_count"] = min(int(quick["scenario"]["train_count"]), 8)
    quick["scenario"]["val_count"] = min(int(quick["scenario"]["val_count"]), 4)
    quick["scenario"]["task_min"] = min(int(quick["scenario"]["task_min"]), 6)
    quick["scenario"]["task_max"] = min(int(quick["scenario"]["task_max"]), 9)
    quick["training"]["episodes"] = min(int(quick["training"]["episodes"]), 10)
    quick["training"]["bc_epochs"] = min(int(quick["training"]["bc_epochs"]), 1)
    return quick
