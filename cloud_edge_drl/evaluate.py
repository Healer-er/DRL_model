"""统一评测流程。

所有策略都通过同一个 `run_scheduler` 调用环境，最终输出相同格式的 makespan 和 ratio。
这样可以公平比较 HEFT、随机策略、RL 策略以及用户自定义策略。
"""

import csv
import os
from typing import Dict, Iterable, List

from .env import SchedulingEnv
from .metrics import rank_by_ratio, summarize
from .policies.heft import HEFTScheduler
from .policies.registry import build_scheduler
from .utils import write_json


def run_scheduler(scenario, scheduler) -> Dict[str, object]:
    """在一个场景上完整运行某个调度器。"""
    env = SchedulingEnv(scenario)
    scheduler.reset(scenario)
    steps = 0
    while not env.done:
        action = scheduler.choose_action(env)
        # env.step 会检查动作合法性，非法自定义策略会在这里暴露问题。
        env.step(action)
        steps += 1
    return {
        "makespan": env.makespan,
        "steps": steps,
        "timeline": env.timeline,
    }


def _heft_makespans(scenarios) -> Dict[str, float]:
    """预先计算每个验证场景的 HEFT makespan，作为 ratio 分母。"""
    result = {}
    for scenario in scenarios:
        output = run_scheduler(scenario, HEFTScheduler())
        result[scenario.scenario_id] = output["makespan"]
    return result


def evaluate(
    scenarios,
    policy_specs: Iterable[str],
    model_path: str,
    result_dir: str,
    seed: int = 0,
) -> Dict[str, object]:
    """评测一组策略并写出 summary.json 和 details.csv。"""
    os.makedirs(result_dir, exist_ok=True)
    heft_by_scenario = _heft_makespans(scenarios)
    records: List[Dict[str, object]] = []

    for spec in policy_specs:
        # 每个策略只构造一次；调度每个场景前会调用 scheduler.reset。
        scheduler = build_scheduler(spec, model_path=model_path, seed=seed)
        for scenario in scenarios:
            output = run_scheduler(scenario, scheduler)
            heft = heft_by_scenario[scenario.scenario_id]
            records.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "policy": scheduler.name,
                    "makespan": output["makespan"],
                    "heft_makespan": heft,
                    "ratio_to_heft": output["makespan"] / heft if heft > 0 else 0.0,
                    "steps": output["steps"],
                }
            )

    # summary 用于排行榜和报告，details 用于逐场景误差分析。
    summary = summarize(records)
    payload = {
        "summary": summary,
        "ranked_policies": rank_by_ratio(summary),
        "records": records,
    }

    write_json(os.path.join(result_dir, "summary.json"), payload)
    write_details_csv(os.path.join(result_dir, "details.csv"), records)
    return payload


def write_details_csv(path: str, records: List[Dict[str, object]]) -> None:
    """把逐场景结果写成表格，方便 Excel 或脚本继续分析。"""
    fields = ["scenario_id", "policy", "makespan", "heft_makespan", "ratio_to_heft", "steps"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record[field] for field in fields})
