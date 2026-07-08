"""Train on the default benchmark and evaluate on harder held-out scenarios."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cloud_edge_drl.config import ensure_output_dirs, load_config
from cloud_edge_drl.evaluate import evaluate
from cloud_edge_drl.scenario import generate_and_save
from cloud_edge_drl.train import train_rl
from cloud_edge_drl.utils import mean


def _paths(config):
    return {
        "model": os.path.join(config["paths"]["model_dir"], "rl_policy.json"),
        "log": os.path.join(config["paths"]["artifact_dir"], "training_log.json"),
    }


def _scenario_stats(scenarios):
    task_counts = [scenario.task_count for scenario in scenarios]
    edge_counts = [len(scenario.edges) for scenario in scenarios]
    resource_counts = [scenario.resource_count for scenario in scenarios]
    return {
        "count": len(scenarios),
        "mean_tasks": mean(task_counts),
        "max_tasks": max(task_counts) if task_counts else 0,
        "mean_edges": mean(edge_counts),
        "max_edges": max(edge_counts) if edge_counts else 0,
        "mean_resources": mean(resource_counts),
    }


def main():
    train_config = load_config(os.path.join(ROOT, "configs", "default.json"))
    complex_config = load_config(os.path.join(ROOT, "configs", "complex.json"))

    ensure_output_dirs(train_config)
    ensure_output_dirs(complex_config)

    train_scenarios, _default_val = generate_and_save(train_config)
    default_paths = _paths(train_config)
    train_rl(train_config, train_scenarios, default_paths["model"], default_paths["log"])

    # Persist the complex split for repeatable inspection without training on it.
    _complex_train, persisted_complex_val = generate_and_save(complex_config)
    complex_val = persisted_complex_val

    stats = _scenario_stats(complex_val)
    print(
        "complex scenarios count=%d mean_tasks=%.2f max_tasks=%d mean_edges=%.2f max_edges=%d mean_resources=%.2f"
        % (
            stats["count"],
            stats["mean_tasks"],
            stats["max_tasks"],
            stats["mean_edges"],
            stats["max_edges"],
            stats["mean_resources"],
        )
    )

    payload = evaluate(
        complex_val,
        complex_config["evaluation"]["policies"],
        model_path=default_paths["model"],
        result_dir=complex_config["paths"]["result_dir"],
        seed=int(complex_config["seed"]),
    )
    print("summary=%s" % os.path.join(complex_config["paths"]["result_dir"], "summary.json"))
    for policy, metrics in payload["summary"].items():
        print(
            "%s mean_ratio=%.4f std_ratio=%.4f mean_makespan=%.4f count=%d"
            % (
                policy,
                metrics["mean_ratio"],
                metrics["std_ratio"],
                metrics["mean_makespan"],
                metrics["count"],
            )
        )


if __name__ == "__main__":
    main()
