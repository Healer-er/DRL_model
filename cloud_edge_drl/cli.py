"""命令行入口。

该模块提供 `generate`、`train`、`evaluate`、`run-all` 四类命令。评审可以直接运行
`python run_all.py --config configs/default.json`，也可以分阶段运行每个子命令。
"""

import argparse
import os
from typing import Any, Dict

from .config import apply_quick_mode, ensure_output_dirs, load_config
from .evaluate import evaluate
from .scenario import generate_and_save, load_scenarios, scenario_paths
from .train import train_rl


def _paths(config: Dict[str, Any]) -> Dict[str, str]:
    """集中生成常用输出路径，避免各命令重复拼接。"""
    return {
        "model": os.path.join(config["paths"]["model_dir"], "rl_policy.json"),
        "log": os.path.join(config["paths"]["artifact_dir"], "training_log.json"),
        "result": config["paths"]["result_dir"],
    }


def _quick_enabled(args) -> bool:
    return bool(getattr(args, "quick", False))


def command_generate(args) -> int:
    """生成并保存训练/验证场景。"""
    config = load_config(args.config)
    if _quick_enabled(args):
        config = apply_quick_mode(config)
    ensure_output_dirs(config)
    train, val = generate_and_save(config)
    print("generated train=%d val=%d" % (len(train), len(val)))
    return 0


def command_train(args) -> int:
    """读取训练场景并训练 RL 策略。"""
    config = load_config(args.config)
    if _quick_enabled(args):
        config = apply_quick_mode(config)
    ensure_output_dirs(config)
    train_path, _val_path = scenario_paths(config)
    if not os.path.exists(train_path):
        # 如果用户直接运行 train，自动补生成数据，减少手工步骤。
        generate_and_save(config)
    train_scenarios = load_scenarios(train_path)
    paths = _paths(config)
    train_rl(config, train_scenarios, paths["model"], paths["log"])
    print("model=%s" % paths["model"])
    return 0


def command_evaluate(args) -> int:
    """在验证集上评测指定策略列表。"""
    config = load_config(args.config)
    if _quick_enabled(args):
        config = apply_quick_mode(config)
    ensure_output_dirs(config)
    train_path, val_path = scenario_paths(config)
    if not os.path.exists(val_path):
        # 如果验证集不存在，自动生成训练/验证场景。
        generate_and_save(config)
    paths = _paths(config)
    policies = args.policies.split(",") if args.policies else config["evaluation"]["policies"]
    payload = evaluate(
        load_scenarios(val_path),
        policies,
        model_path=args.model or paths["model"],
        result_dir=paths["result"],
        seed=int(config["seed"]),
    )
    print("summary=%s" % os.path.join(paths["result"], "summary.json"))
    for policy, metrics in payload["summary"].items():
        print("%s mean_ratio=%.4f count=%d" % (policy, metrics["mean_ratio"], metrics["count"]))
    return 0


def command_run_all(args) -> int:
    """一键执行生成、训练和评测，是提交给评审的主入口。"""
    config = load_config(args.config)
    if _quick_enabled(args):
        config = apply_quick_mode(config)
    ensure_output_dirs(config)
    train, val = generate_and_save(config)
    paths = _paths(config)
    train_rl(config, train, paths["model"], paths["log"])
    payload = evaluate(
        val,
        config["evaluation"]["policies"],
        model_path=paths["model"],
        result_dir=paths["result"],
        seed=int(config["seed"]),
    )
    print("summary=%s" % os.path.join(paths["result"], "summary.json"))
    for policy, metrics in payload["summary"].items():
        print("%s mean_ratio=%.4f std_ratio=%.4f count=%d" % (
            policy,
            metrics["mean_ratio"],
            metrics["std_ratio"],
            metrics["count"],
        ))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。

    common parser 同时挂到主命令和子命令上，所以 `--config` 放在子命令前后都能识别。
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="configs/default.json", help="Path to JSON config")
    common.add_argument(
        "--quick",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Run a tiny smoke-sized experiment",
    )

    parser = argparse.ArgumentParser(
        description="Cloud-edge-device DRL scheduler framework",
        parents=[common],
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("generate", parents=[common])
    subparsers.add_parser("train", parents=[common])

    evaluate_parser = subparsers.add_parser("evaluate", parents=[common])
    evaluate_parser.add_argument("--policies", default=None, help="Comma separated policies")
    evaluate_parser.add_argument("--model", default=None, help="RL model path")

    subparsers.add_parser("run-all", parents=[common])
    return parser


def main(argv=None) -> int:
    """CLI 主函数；未指定子命令时默认执行 run-all。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "run-all"
    if command == "generate":
        return command_generate(args)
    if command == "train":
        return command_train(args)
    if command == "evaluate":
        return command_evaluate(args)
    if command == "run-all":
        return command_run_all(args)
    parser.error("unknown command %s" % command)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
