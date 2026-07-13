"""Generate a Markdown report and SVG Gantt charts for scheduler results.

The script reruns selected policies on the validation scenarios so it can
capture timelines without changing the main evaluator output format. It uses
only the Python standard library.
"""

import argparse
import csv
import html
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cloud_edge_drl.config import ensure_output_dirs, load_config
from cloud_edge_drl.evaluate import run_scheduler
from cloud_edge_drl.metrics import rank_by_ratio, summarize
from cloud_edge_drl.policies.registry import build_scheduler
from cloud_edge_drl.scenario import generate_and_save, load_scenarios, scenario_paths
from cloud_edge_drl.utils import write_json


KIND_COLORS = {
    "cloud": "#2563eb",
    "edge": "#059669",
    "device": "#d97706",
}


def _default_model_path(config):
    return os.path.join(config["paths"]["model_dir"], "rl_policy.json")


def _load_validation_scenarios(config):
    _train_path, val_path = scenario_paths(config)
    if not os.path.exists(val_path):
        generate_and_save(config)
    return load_scenarios(val_path)


def _evaluate_with_timelines(scenarios, policies, model_path, seed):
    heft_by_scenario = {}
    records = []
    timelines = {}

    for scenario in scenarios:
        heft_output = run_scheduler(scenario, build_scheduler("heft", model_path=model_path, seed=seed))
        heft_by_scenario[scenario.scenario_id] = heft_output["makespan"]

    for spec in policies:
        scheduler = build_scheduler(spec, model_path=model_path, seed=seed)
        for scenario in scenarios:
            output = run_scheduler(scenario, scheduler)
            heft = heft_by_scenario[scenario.scenario_id]
            record = {
                "scenario_id": scenario.scenario_id,
                "policy": scheduler.name,
                "makespan": output["makespan"],
                "heft_makespan": heft,
                "ratio_to_heft": output["makespan"] / heft if heft > 0 else 0.0,
                "steps": output["steps"],
            }
            records.append(record)
            timelines[(scenario.scenario_id, scheduler.name)] = output["timeline"]
    return records, timelines


def _write_details_csv(path, records):
    fields = ["scenario_id", "policy", "makespan", "heft_makespan", "ratio_to_heft", "steps"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record[field] for field in fields})


def _pick_report_scenario(records, requested=None):
    if requested:
        return requested
    candidates = [record for record in records if record["policy"] != "heft"]
    if not candidates:
        return records[0]["scenario_id"]
    best = min(candidates, key=lambda item: (item["ratio_to_heft"], item["scenario_id"], item["policy"]))
    return best["scenario_id"]


def _svg_gantt(scenario, policy, timeline, path):
    max_finish = max((item["finish"] for item in timeline), default=1.0)
    left = 120
    right = 28
    top = 48
    row_h = 44
    chart_w = 900
    height = top + row_h * len(scenario.resources) + 48
    width = left + chart_w + right
    scale = chart_w / max(max_finish, 1.0)
    row_index = {resource.resource_id: index for index, resource in enumerate(scenario.resources)}

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d">'
        % (width, height, width, height),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="28" font-family="Arial" font-size="18" font-weight="700" fill="#111827">%s - %s</text>'
        % (html.escape(scenario.scenario_id), html.escape(policy)),
        '<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#9ca3af" stroke-width="1"/>'
        % (left, top - 10, left + chart_w, top - 10),
    ]

    for resource in scenario.resources:
        y = top + row_index[resource.resource_id] * row_h
        color = KIND_COLORS.get(resource.kind, "#6b7280")
        lines.append(
            '<text x="24" y="%d" font-family="Arial" font-size="13" fill="#111827">%s (%s)</text>'
            % (y + 25, html.escape(resource.resource_id), html.escape(resource.kind))
        )
        lines.append(
            '<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#e5e7eb" stroke-width="1"/>'
            % (left, y + row_h - 4, left + chart_w, y + row_h - 4)
        )
        for tick in range(6):
            x = left + int(chart_w * tick / 5.0)
            t = max_finish * tick / 5.0
            lines.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#f3f4f6" stroke-width="1"/>' % (x, top - 18, x, height - 34))
            lines.append('<text x="%d" y="%d" font-family="Arial" font-size="10" fill="#6b7280">%.1f</text>' % (x - 10, height - 16, t))
        lines.append(
            '<rect x="%d" y="%d" width="10" height="10" rx="2" fill="%s"/>'
            % (left - 18, y + 17, color)
        )

    for item in sorted(timeline, key=lambda value: (value["resource"], value["start"], value["task"])):
        resource = scenario.resource_by_id[item["resource"]]
        color = KIND_COLORS.get(resource.kind, "#6b7280")
        y = top + row_index[item["resource"]] * row_h + 9
        x = left + item["start"] * scale
        w = max(2.0, (item["finish"] - item["start"]) * scale)
        label = "T%s" % item["task"]
        lines.append(
            '<rect x="%.2f" y="%d" width="%.2f" height="24" rx="4" fill="%s" opacity="0.86"/>'
            % (x, y, w, color)
        )
        if w >= 24:
            lines.append(
                '<text x="%.2f" y="%d" font-family="Arial" font-size="11" fill="#ffffff">%s</text>'
                % (x + 5, y + 16, html.escape(label))
            )
    lines.append("</svg>")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _write_markdown_report(path, config_path, scenario_id, summary, ranked, gantt_files):
    lines = [
        "# Scheduler Visual Report",
        "",
        "- config: `%s`" % config_path,
        "- selected scenario: `%s`" % scenario_id,
        "- metric: `ratio_to_heft = policy_makespan / HEFT_makespan`",
        "",
        "## Policy Ranking",
        "",
        "| rank | policy | count | mean_ratio | std_ratio | mean_makespan |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for index, policy in enumerate(ranked, start=1):
        item = summary[policy]
        lines.append(
            "| %d | %s | %d | %.6f | %.6f | %.6f |"
            % (
                index,
                policy,
                item["count"],
                item["mean_ratio"],
                item["std_ratio"],
                item["mean_makespan"],
            )
        )
    lines.extend(["", "## Gantt Charts", ""])
    for policy, filename in gantt_files:
        lines.append("### %s" % policy)
        lines.append("")
        lines.append("![%s](%s)" % (policy, filename))
        lines.append("")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate scheduler visual report")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--model", default=None)
    parser.add_argument("--policies", default=None, help="Comma-separated policy specs")
    parser.add_argument("--scenario-id", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    ensure_output_dirs(config)
    scenarios = _load_validation_scenarios(config)
    policies = args.policies.split(",") if args.policies else config["evaluation"]["policies"]
    model_path = args.model or _default_model_path(config)
    output_dir = args.output_dir or os.path.join(config["paths"]["result_dir"], "visual_report")
    os.makedirs(output_dir, exist_ok=True)

    records, timelines = _evaluate_with_timelines(scenarios, policies, model_path, int(config["seed"]))
    summary = summarize(records)
    ranked = rank_by_ratio(summary)
    scenario_id = _pick_report_scenario(records, args.scenario_id)
    scenario_by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    if scenario_id not in scenario_by_id:
        raise ValueError("Unknown scenario id: %s" % scenario_id)

    gantt_files = []
    scenario = scenario_by_id[scenario_id]
    for policy in ranked:
        timeline = timelines.get((scenario_id, policy))
        if timeline is None:
            continue
        filename = "gantt_%s_%s.svg" % (scenario_id, policy)
        _svg_gantt(scenario, policy, timeline, os.path.join(output_dir, filename))
        gantt_files.append((policy, filename))

    payload = {"summary": summary, "ranked_policies": ranked, "records": records}
    write_json(os.path.join(output_dir, "summary.json"), payload)
    _write_details_csv(os.path.join(output_dir, "details.csv"), records)
    write_json(
        os.path.join(output_dir, "timelines.json"),
        [
            {"scenario_id": key[0], "policy": key[1], "timeline": value}
            for key, value in sorted(timelines.items())
        ],
    )
    _write_markdown_report(
        os.path.join(output_dir, "report.md"),
        args.config,
        scenario_id,
        summary,
        ranked,
        gantt_files,
    )
    print("report=%s" % os.path.join(output_dir, "report.md"))
    print("charts=%d" % len(gantt_files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
