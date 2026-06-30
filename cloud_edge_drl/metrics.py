"""评测指标聚合。

核心指标是 `mean_ratio = mean(policy_makespan / HEFT_makespan)`，同时报告标准差和样本数，
满足赛题对均值、方差或样本规模说明的要求。
"""

from typing import Dict, Iterable, List

from .utils import mean, stddev


def summarize(records: Iterable[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    """按策略聚合逐场景记录。"""
    grouped = {}
    for record in records:
        grouped.setdefault(record["policy"], []).append(record)

    summary = {}
    for policy, items in grouped.items():
        # makespan 衡量绝对完成时间，ratio 衡量相对 HEFT 的标准化性能。
        makespans = [float(item["makespan"]) for item in items]
        ratios = [float(item["ratio_to_heft"]) for item in items]
        summary[policy] = {
            "count": len(items),
            "mean_makespan": mean(makespans),
            "std_makespan": stddev(makespans),
            "mean_ratio": mean(ratios),
            "std_ratio": stddev(ratios),
        }
    return summary


def rank_by_ratio(summary: Dict[str, Dict[str, float]]) -> List[str]:
    """按 mean_ratio 从小到大排序策略名。"""
    return sorted(summary, key=lambda name: summary[name]["mean_ratio"])
