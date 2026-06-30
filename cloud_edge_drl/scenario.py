"""场景生成与加载。

赛题通常会给出或要求生成多组验证场景。本模块负责把“任务图 + 资源参数”
生成成统一的 `Scenario` 对象，并保存为 JSON。训练集和验证集使用不同随机种子偏移，
避免同一批场景同时参与训练和评估。
"""

import json
import os
import random
from typing import Any, Dict, Iterable, List, Tuple

from .domain import Edge, Resource, Scenario, Task


def _uniform_pair(rng: random.Random, bounds: Iterable[float]) -> float:
    """从 `[low, high]` 区间均匀采样一个浮点数。"""
    low, high = list(bounds)
    return rng.uniform(float(low), float(high))


def _make_resources(config: Dict[str, Any], rng: random.Random) -> List[Resource]:
    """根据配置生成云、边、端三类资源。

    资源 id 使用 `c/e/d + 编号`，便于结果文件中直接看出任务被分配到哪一层。
    """
    resources = []
    prefixes = {"cloud": "c", "edge": "e", "device": "d"}
    for kind in ("cloud", "edge", "device"):
        count = int(config["resources"]["counts"].get(kind, 0))
        speed_range = config["resources"]["speed_ranges"][kind]
        for index in range(count):
            speed = round(_uniform_pair(rng, speed_range), 4)
            resources.append(Resource(resource_id="%s%d" % (prefixes[kind], index), kind=kind, speed=speed))
    return resources


def _make_bandwidth(config: Dict[str, Any], resources: List[Resource], rng: random.Random) -> Dict[str, Dict[str, float]]:
    """生成资源到资源的带宽矩阵。

    配置文件给出的是按资源类型聚合的基础带宽；这里加入少量 jitter，让同类型资源
    也存在细微差异，更接近异构系统。
    """
    by_kind = config["resources"]["bandwidth_by_kind"]
    matrix = {}
    for src in resources:
        matrix[src.resource_id] = {}
        for dst in resources:
            key = "%s-%s" % (src.kind, dst.kind)
            base = float(by_kind[key])
            jitter = rng.uniform(0.92, 1.08)
            matrix[src.resource_id][dst.resource_id] = round(base * jitter, 4)
    return matrix


def generate_scenario(config: Dict[str, Any], rng: random.Random, scenario_id: str) -> Scenario:
    """随机生成一个 DAG 调度场景。

    生成逻辑保持简单透明：任务 id 的自然顺序就是拓扑序，只允许从小 id 指向大 id。
    这保证生成结果一定无环，同时仍然可以通过 `edge_probability` 控制依赖稠密度。
    """
    scenario_config = config["scenario"]
    task_min = int(scenario_config["task_min"])
    task_max = int(scenario_config["task_max"])
    task_count = rng.randint(task_min, task_max)

    tasks = [
        Task(task_id=index, work=round(_uniform_pair(rng, scenario_config["work_range"]), 4))
        for index in range(task_count)
    ]

    edges = []
    edge_probability = float(scenario_config["edge_probability"])
    for child in range(1, task_count):
        parents = []
        for parent in range(child):
            # 距离较近的任务略微提高连边概率，形成更自然的局部依赖结构。
            distance_boost = 1.0 + 0.3 * (1.0 - float(child - parent) / float(task_count))
            if rng.random() < min(0.75, edge_probability * distance_boost):
                parents.append(parent)
        # 偶尔强制给任务添加一个父节点，避免 DAG 退化成大量完全独立任务。
        if not parents and rng.random() < 0.65:
            parents.append(rng.randrange(0, child))
        for parent in sorted(set(parents)):
            data = round(_uniform_pair(rng, scenario_config["data_range"]), 4)
            edges.append(Edge(parent=parent, child=child, data=data))

    resources = _make_resources(config, rng)
    bandwidth = _make_bandwidth(config, resources, rng)
    return Scenario(scenario_id=scenario_id, tasks=tasks, edges=edges, resources=resources, bandwidth=bandwidth)


def generate_dataset(config: Dict[str, Any], split: str) -> List[Scenario]:
    """生成训练集或验证集。

    验证集在全局 seed 上增加一个大偏移，确保与训练集可复现但不重合。
    """
    seed = int(config["seed"]) + (0 if split == "train" else 100000)
    rng = random.Random(seed)
    count_key = "train_count" if split == "train" else "val_count"
    count = int(config["scenario"][count_key])
    return [generate_scenario(config, rng, "%s_%03d" % (split, index)) for index in range(count)]


def save_scenarios(path: str, scenarios: Iterable[Scenario]) -> None:
    """保存场景列表到 JSON 文件。"""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump([scenario.to_dict() for scenario in scenarios], handle, indent=2, ensure_ascii=False)


def load_scenarios(path: str) -> List[Scenario]:
    """从 JSON 文件读取场景列表。"""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return [Scenario.from_dict(item) for item in data]


def scenario_paths(config: Dict[str, Any]) -> Tuple[str, str]:
    """根据配置返回训练集和验证集路径。"""
    scenario_dir = config["paths"]["scenario_dir"]
    return os.path.join(scenario_dir, "train.json"), os.path.join(scenario_dir, "val.json")


def generate_and_save(config: Dict[str, Any]) -> Tuple[List[Scenario], List[Scenario]]:
    """生成并保存训练/验证两份场景。"""
    train_path, val_path = scenario_paths(config)
    train = generate_dataset(config, "train")
    val = generate_dataset(config, "val")
    save_scenarios(train_path, train)
    save_scenarios(val_path, val)
    return train, val
