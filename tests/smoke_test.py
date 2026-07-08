"""不依赖第三方测试框架的最小自检。

该脚本使用 quick 配置跑通“生成场景 -> 训练 -> 评测”全链路，适合在新系统上
快速确认代码和 Python 环境没有问题。
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cloud_edge_drl.config import apply_quick_mode, ensure_output_dirs, load_config
from cloud_edge_drl.evaluate import evaluate
from cloud_edge_drl.scenario import generate_dataset
from cloud_edge_drl.train import train_rl


def main():
    """执行烟测并检查 summary 中存在必要策略指标。"""
    config = apply_quick_mode(load_config(os.path.join(ROOT, "configs", "default.json")))
    ensure_output_dirs(config)
    train = generate_dataset(config, "train")
    val = generate_dataset(config, "val")
    model_path = os.path.join(config["paths"]["model_dir"], "smoke_rl_policy.json")
    log_path = os.path.join(config["paths"]["artifact_dir"], "smoke_training_log.json")
    train_rl(config, train, model_path, log_path)
    payload = evaluate(val, ["heft", "lookahead_heft", "rl"], model_path, config["paths"]["result_dir"], seed=config["seed"])
    assert "heft" in payload["summary"]
    assert "lookahead_heft" in payload["summary"]
    assert "rl" in payload["summary"]
    assert payload["summary"]["heft"]["mean_ratio"] == 1.0
    assert payload["summary"]["lookahead_heft"]["mean_ratio"] <= 1.0
    assert payload["summary"]["rl"]["mean_ratio"] <= 1.0
    print("smoke test passed")


if __name__ == "__main__":
    main()
