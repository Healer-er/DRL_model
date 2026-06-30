"""训练和评测共用的策略接口。

赛题强调“统一调度策略接口”，因此所有策略都只需要实现 `choose_action(env)`。
评测脚本不需要知道策略内部是启发式、行为克隆还是强化学习模型。
"""


class Scheduler:
    """调度器基类。

    子类可以继承它，也可以只实现同名方法。`name` 会写入结果文件，用于区分策略。
    """

    name = "scheduler"

    def reset(self, scenario):
        """每个新场景开始前调用。

        HEFT 这类策略会在这里预计算 upward rank；无状态策略可以留空。
        """

    def choose_action(self, env):
        """返回一个合法的 `(task_id, resource_id)` 动作。"""
        raise NotImplementedError
