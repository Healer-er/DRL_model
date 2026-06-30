"""核心数据结构。

这个模块只描述“场景是什么”，不包含任何调度算法。这样做的好处是：
场景可以被环境、HEFT、RL 策略、评测脚本共同复用，评审替换数据来源时也只需要
生成同样结构的 `Scenario` 对象即可。
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class Task:
    """DAG 中的一个任务节点。

    `work` 表示任务计算量。真实系统中它可以对应 CPU cycles、指令量、
    或者评测方定义的抽象计算负载。
    """

    task_id: int
    work: float

    def to_dict(self) -> Dict[str, float]:
        """转成 JSON 友好的字典，便于保存验证集和复现实验。"""
        return {"id": self.task_id, "work": self.work}

    @staticmethod
    def from_dict(data: Dict[str, float]) -> "Task":
        """从 JSON 字典恢复任务对象。"""
        return Task(task_id=int(data["id"]), work=float(data["work"]))


@dataclass(frozen=True)
class Edge:
    """DAG 中的一条依赖边。

    `parent -> child` 表示 child 必须等待 parent 完成。`data` 表示父子任务之间
    需要传输的数据量；如果两个任务落在不同计算资源上，环境会据此计算通信时间。
    """

    parent: int
    child: int
    data: float

    def to_dict(self) -> Dict[str, float]:
        """转成可写入场景文件的字典。"""
        return {"parent": self.parent, "child": self.child, "data": self.data}

    @staticmethod
    def from_dict(data: Dict[str, float]) -> "Edge":
        """从场景文件中的字典恢复依赖边。"""
        return Edge(parent=int(data["parent"]), child=int(data["child"]), data=float(data["data"]))


@dataclass(frozen=True)
class Resource:
    """一个计算资源节点。

    `kind` 用来区分 cloud / edge / device；`speed` 表示处理速度。调度算法可以
    使用 `kind` 表示资源层级，也可以只把它当作特征交给学习策略。
    """

    resource_id: str
    kind: str
    speed: float

    def to_dict(self) -> Dict[str, float]:
        """转成 JSON 友好的资源描述。"""
        return {"id": self.resource_id, "kind": self.kind, "speed": self.speed}

    @staticmethod
    def from_dict(data: Dict[str, float]) -> "Resource":
        """从 JSON 字典恢复资源对象。"""
        return Resource(resource_id=str(data["id"]), kind=str(data["kind"]), speed=float(data["speed"]))


class Scenario:
    """一个完整调度场景。

    它包含任务 DAG、异构资源列表和资源间带宽矩阵。类初始化时会预先建立
    `parents`、`children`、`edge_data` 等索引，后续环境仿真和策略特征提取
    可以直接查询，避免每一步重复扫描边列表。
    """

    def __init__(
        self,
        scenario_id: str,
        tasks: Iterable[Task],
        edges: Iterable[Edge],
        resources: Iterable[Resource],
        bandwidth: Dict[str, Dict[str, float]],
    ) -> None:
        self.scenario_id = scenario_id
        self.tasks = sorted(list(tasks), key=lambda item: item.task_id)
        self.edges = list(edges)
        self.resources = list(resources)
        self.bandwidth = bandwidth

        self.task_by_id = {task.task_id: task for task in self.tasks}
        self.resource_by_id = {resource.resource_id: resource for resource in self.resources}
        # parents/children 是 ready task 判断和 HEFT upward rank 的核心索引。
        self.parents = {task.task_id: [] for task in self.tasks}
        self.children = {task.task_id: [] for task in self.tasks}
        # edge_data[(u, v)] 记录 u 到 v 的数据量，用于跨资源通信时间估计。
        self.edge_data = {}
        for edge in self.edges:
            self.parents[edge.child].append(edge.parent)
            self.children[edge.parent].append(edge.child)
            self.edge_data[(edge.parent, edge.child)] = edge.data
        for task_id in self.parents:
            self.parents[task_id].sort()
            self.children[task_id].sort()

        self._validate()

    def _validate(self) -> None:
        """做轻量一致性检查，尽早暴露坏场景文件。"""
        task_ids = set(self.task_by_id)
        if len(task_ids) != len(self.tasks):
            raise ValueError("Task ids must be unique.")
        if not self.resources:
            raise ValueError("Scenario must contain at least one resource.")
        for edge in self.edges:
            if edge.parent not in task_ids or edge.child not in task_ids:
                raise ValueError("Every edge endpoint must reference an existing task.")
            # 生成器使用 task_id 的自然顺序作为拓扑序，因此父节点 id 必须小于子节点。
            if edge.parent >= edge.child:
                raise ValueError("Generated scenarios use task id order as topological order.")
        for resource in self.resources:
            if resource.speed <= 0:
                raise ValueError("Resource speed must be positive.")
        for src in self.resource_by_id:
            if src not in self.bandwidth:
                raise ValueError("Bandwidth matrix misses source resource %s." % src)
            for dst in self.resource_by_id:
                if dst not in self.bandwidth[src] or self.bandwidth[src][dst] <= 0:
                    raise ValueError("Bandwidth matrix must be positive for every resource pair.")

    @property
    def task_count(self) -> int:
        """任务数量，常用于判断 episode 是否结束。"""
        return len(self.tasks)

    @property
    def resource_count(self) -> int:
        """资源数量，动作空间大小等于 ready task 数乘以该值。"""
        return len(self.resources)

    def task_ids(self) -> List[int]:
        return [task.task_id for task in self.tasks]

    def resource_ids(self) -> List[str]:
        return [resource.resource_id for resource in self.resources]

    def max_work(self) -> float:
        """最大任务计算量，用于特征归一化。"""
        return max(task.work for task in self.tasks)

    def max_speed(self) -> float:
        """最快资源速度，用于计算理想计算下界和归一化。"""
        return max(resource.speed for resource in self.resources)

    def avg_speed(self) -> float:
        """平均资源速度，HEFT upward rank 中用它估计平均执行代价。"""
        return sum(resource.speed for resource in self.resources) / float(len(self.resources))

    def avg_bandwidth(self) -> float:
        """非同源资源之间的平均带宽，供启发式 rank 估计平均通信代价。"""
        values = []
        for src, row in self.bandwidth.items():
            for dst, value in row.items():
                if src != dst:
                    values.append(value)
        return sum(values) / float(len(values)) if values else 1.0

    def communication_time(self, parent: int, child: int, src_resource: str, dst_resource: str) -> float:
        """计算父子任务因部署在不同资源上产生的通信时间。"""
        if src_resource == dst_resource:
            # 同一资源内部传递数据视为本地访问，不产生跨节点网络传输时间。
            return 0.0
        data = self.edge_data.get((parent, child), 0.0)
        return data / self.bandwidth[src_resource][dst_resource]

    def to_dict(self) -> Dict[str, object]:
        """完整序列化一个场景，保证训练/验证划分可落盘复现。"""
        return {
            "id": self.scenario_id,
            "tasks": [task.to_dict() for task in self.tasks],
            "edges": [edge.to_dict() for edge in self.edges],
            "resources": [resource.to_dict() for resource in self.resources],
            "bandwidth": self.bandwidth,
        }

    @staticmethod
    def from_dict(data: Dict[str, object]) -> "Scenario":
        """从场景 JSON 恢复 Scenario 对象。"""
        return Scenario(
            scenario_id=str(data["id"]),
            tasks=[Task.from_dict(item) for item in data["tasks"]],
            edges=[Edge.from_dict(item) for item in data["edges"]],
            resources=[Resource.from_dict(item) for item in data["resources"]],
            bandwidth={
                str(src): {str(dst): float(value) for dst, value in row.items()}
                for src, row in data["bandwidth"].items()
            },
        )

    def topological_edges(self) -> List[Tuple[int, int]]:
        """返回边的 `(parent, child)` 形式，便于调试或绘制 DAG。"""
        return [(edge.parent, edge.child) for edge in self.edges]
