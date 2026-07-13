# 可视化报告与甘特图使用说明

本项目新增 `scripts/generate_visual_report.py`，用于把调度评测结果转换为更适合答辩展示的 Markdown 报告和 SVG 甘特图。脚本只依赖 Python 标准库。

## 运行命令

默认配置：

```bash
python scripts/generate_visual_report.py --config configs/default.json
```

复杂配置：

```bash
python scripts/generate_visual_report.py --config configs/complex.json --output-dir results/complex/visual_report
```

指定策略集合：

```bash
python scripts/generate_visual_report.py ^
  --config configs/default.json ^
  --policies heft,lookahead_heft,rl_bc_only,rl_greedy,rl
```

指定要绘制甘特图的验证场景：

```bash
python scripts/generate_visual_report.py ^
  --config configs/default.json ^
  --scenario-id val_003
```

## 输出文件

默认输出目录为：

```text
results/visual_report/
```

主要产物：

- `report.md`：策略排名表和甘特图索引。
- `summary.json`：策略聚合指标。
- `details.csv`：逐场景、逐策略结果。
- `timelines.json`：每个场景和策略的完整任务时间线。
- `gantt_<scenario>_<policy>.svg`：单场景、单策略甘特图。

## 答辩使用方式

建议在答辩材料中同时放两类内容：

- `report.md` 中的策略排名表，用于说明整体性能。
- `gantt_<scenario>_heft.svg` 与 `gantt_<scenario>_rl.svg`，用于直观看到 RL/rollout-safe 策略如何缩短资源空闲、改变任务放置或降低 makespan。

如果需要突出消融实验，可以同时展示：

- `gantt_<scenario>_rl_bc_only.svg`
- `gantt_<scenario>_rl_greedy.svg`
- `gantt_<scenario>_rl.svg`

这样可以解释行为克隆、纯 MLP 贪心和安全 rollout 推理之间的差异。
