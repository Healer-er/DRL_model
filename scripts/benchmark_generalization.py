"""Run multi-seed generalization benchmarks and write an aggregate report.

This script is intended for competition evidence, not quick smoke testing. It
trains and evaluates the same policy set under multiple random seeds, then
aggregates mean_ratio statistics across all validation scenarios.
"""

import argparse
import json
import os
import sys
from copy import deepcopy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cloud_edge_drl.config import ensure_output_dirs, load_config
from cloud_edge_drl.evaluate import evaluate
from cloud_edge_drl.scenario import generate_dataset
from cloud_edge_drl.train import train_rl
from cloud_edge_drl.utils import mean, stddev, write_json


def _parse_seeds(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _with_run_paths(config, output_root, seed):
    run_config = deepcopy(config)
    run_config["seed"] = int(seed)
    run_root = os.path.join(output_root, "seed_%s" % seed)
    run_config["paths"]["artifact_dir"] = os.path.join(run_root, "artifacts")
    run_config["paths"]["scenario_dir"] = os.path.join(run_root, "artifacts", "scenarios")
    run_config["paths"]["model_dir"] = os.path.join(run_root, "artifacts", "models")
    run_config["paths"]["result_dir"] = os.path.join(run_root, "results")
    return run_config


def _aggregate(run_payloads):
    by_policy = {}
    for item in run_payloads:
        seed = item["seed"]
        for record in item["payload"]["records"]:
            policy = record["policy"]
            by_policy.setdefault(policy, []).append(
                {
                    "seed": seed,
                    "scenario_id": record["scenario_id"],
                    "ratio_to_heft": float(record["ratio_to_heft"]),
                    "makespan": float(record["makespan"]),
                }
            )

    summary = {}
    for policy, records in by_policy.items():
        ratios = [record["ratio_to_heft"] for record in records]
        wins = sum(1 for value in ratios if value < 1.0)
        ties = sum(1 for value in ratios if abs(value - 1.0) <= 1e-12)
        summary[policy] = {
            "count": len(records),
            "mean_ratio": mean(ratios),
            "std_ratio": stddev(ratios),
            "best_ratio": min(ratios),
            "worst_ratio": max(ratios),
            "win_rate_vs_heft": wins / float(max(1, len(ratios))),
            "non_worse_rate_vs_heft": (wins + ties) / float(max(1, len(ratios))),
        }
    return summary


def _write_report(path, config_path, seeds, aggregate):
    ranked = sorted(aggregate, key=lambda name: aggregate[name]["mean_ratio"])
    lines = [
        "# Generalization Benchmark Report",
        "",
        "- config: `%s`" % config_path,
        "- seeds: `%s`" % ",".join(str(seed) for seed in seeds),
        "- metric: `ratio_to_heft = policy_makespan / HEFT_makespan`",
        "",
        "| rank | policy | count | mean_ratio | std_ratio | best | worst | win_rate | non_worse_rate |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index, policy in enumerate(ranked, start=1):
        item = aggregate[policy]
        lines.append(
            "| %d | %s | %d | %.6f | %.6f | %.6f | %.6f | %.2f%% | %.2f%% |"
            % (
                index,
                policy,
                item["count"],
                item["mean_ratio"],
                item["std_ratio"],
                item["best_ratio"],
                item["worst_ratio"],
                item["win_rate_vs_heft"] * 100.0,
                item["non_worse_rate_vs_heft"] * 100.0,
            )
        )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run multi-seed scheduler benchmarks")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--seeds", default="2026,2027,2028")
    parser.add_argument("--output-root", default="results/generalization")
    parser.add_argument("--train-count", type=int, default=None)
    parser.add_argument("--val-count", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args(argv)

    base_config = load_config(args.config)
    if args.train_count is not None:
        base_config["scenario"]["train_count"] = args.train_count
    if args.val_count is not None:
        base_config["scenario"]["val_count"] = args.val_count
    if args.episodes is not None:
        base_config["training"]["episodes"] = args.episodes

    seeds = _parse_seeds(args.seeds)
    os.makedirs(args.output_root, exist_ok=True)

    run_payloads = []
    for seed in seeds:
        config = _with_run_paths(base_config, args.output_root, seed)
        ensure_output_dirs(config)
        train = generate_dataset(config, "train")
        val = generate_dataset(config, "val")
        model_path = os.path.join(config["paths"]["model_dir"], "rl_policy.json")
        log_path = os.path.join(config["paths"]["artifact_dir"], "training_log.json")
        train_rl(config, train, model_path, log_path)
        payload = evaluate(
            val,
            config["evaluation"]["policies"],
            model_path=model_path,
            result_dir=config["paths"]["result_dir"],
            seed=seed,
        )
        run_payloads.append({"seed": seed, "payload": payload})
        print("seed=%s best=%s" % (seed, payload["ranked_policies"][0]))

    aggregate = _aggregate(run_payloads)
    output = {
        "config": args.config,
        "seeds": seeds,
        "aggregate": aggregate,
        "runs": [
            {"seed": item["seed"], "summary": item["payload"]["summary"]}
            for item in run_payloads
        ],
    }
    write_json(os.path.join(args.output_root, "aggregate.json"), output)
    _write_report(os.path.join(args.output_root, "report.md"), args.config, seeds, aggregate)
    print("aggregate=%s" % os.path.join(args.output_root, "aggregate.json"))
    print("report=%s" % os.path.join(args.output_root, "report.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
