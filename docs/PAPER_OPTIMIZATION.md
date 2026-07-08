# Paper-Inspired Optimization Notes

The two GrapheonRL papers motivate three practical changes for this project.

## 1. Constraint-Aware Action Selection

The papers model scheduling as task-to-node decisions under dependency, resource, and data-transfer constraints. The existing environment already enforces dependency legality and data-transfer time. We keep that strict action mask and add stronger policy choices on top of it.

The RL policy now uses a `rollout_safe` inference mode. The MLP actor scores legal actions, then the scheduler evaluates the actor's top candidates plus a strong rollout teacher action by completing the remaining DAG with HEFT. This keeps the learned policy in the decision loop while preventing poor greedy choices from dominating validation performance.

## 2. Stronger Heuristic Portfolio

The papers compare learning methods against HEFT and classic heuristics such as OLB/MET-style rules. This project now includes additional deterministic baselines:

- `min_min`: choose the ready task-resource pair with the smallest earliest finish time.
- `max_min`: choose the task whose best resource still has the largest earliest finish time.
- `portfolio`: run HEFT, one-step HEFT rollout, min-min, and max-min, then execute the complete plan with the lowest makespan.

Because `portfolio` includes `lookahead_heft`, it is never worse than that strategy for a deterministic scenario evaluation.

## 3. Harder Graph-Structured Validation

The papers emphasize evaluation on larger and denser workflow graphs. The project now includes `configs/complex.json` and `scripts/evaluate_complex.py`, which test larger DAGs, more resources, higher dependency density, and heavier data-transfer pressure.

This exposes the current lightweight MLP actor's generalization limit without requiring a deep learning framework.

## Deferred Deep-Learning Work

The papers' main model uses GNN embeddings with PPO/critic training. Reproducing that faithfully would require adding a framework such as PyTorch or PyTorch Geometric and changing the dependency footprint. The current implementation intentionally stays standard-library-only, so the implemented changes focus on deterministic graph-aware scheduling and stronger evaluation.
