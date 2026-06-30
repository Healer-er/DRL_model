"""一键训练与验证评测入口。

这个文件保持很薄，只把控制权交给 `cloud_edge_drl.cli`。这样命令行能力集中维护，
同时评审仍然可以用最简单的 `python run_all.py` 启动完整流程。
"""

from cloud_edge_drl.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
