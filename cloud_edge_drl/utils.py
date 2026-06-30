"""训练和评测共用的小工具。

这里不放业务逻辑，只放 JSON 读写、统计量、裁剪等基础函数。拆出来可以让训练、
评测、指标模块保持简洁。
"""

import json
import math
import os
from typing import Iterable, List


def write_json(path: str, data: object) -> None:
    """把对象写成格式化 JSON，并自动创建父目录。"""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def read_json(path: str) -> object:
    """读取 JSON 文件。"""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def mean(values: Iterable[float]) -> float:
    """计算均值；空列表返回 0，避免评测早期调试时崩溃。"""
    data = list(values)
    return sum(data) / float(len(data)) if data else 0.0


def variance(values: Iterable[float]) -> float:
    """计算样本方差，用于报告验证集波动。"""
    data = list(values)
    if len(data) <= 1:
        return 0.0
    mu = mean(data)
    return sum((value - mu) ** 2 for value in data) / float(len(data) - 1)


def stddev(values: Iterable[float]) -> float:
    """计算标准差。"""
    return math.sqrt(variance(values))


def clamp(value: float, limit: float) -> float:
    """把数值限制在 `[-limit, limit]`，训练时用于裁剪 advantage。"""
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


def flatten(list_of_lists: Iterable[Iterable[float]]) -> List[float]:
    """展开二维列表；保留给后续批处理特征或日志分析使用。"""
    result = []
    for values in list_of_lists:
        result.extend(values)
    return result
