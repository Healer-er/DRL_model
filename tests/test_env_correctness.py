"""Core correctness tests for the scheduling environment."""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cloud_edge_drl.domain import Edge, Resource, Scenario, Task
from cloud_edge_drl.env import SchedulingEnv


def make_two_task_scenario():
    resources = [
        Resource(resource_id="c0", kind="cloud", speed=10.0),
        Resource(resource_id="d0", kind="device", speed=2.0),
    ]
    bandwidth = {
        "c0": {"c0": 100.0, "d0": 5.0},
        "d0": {"c0": 5.0, "d0": 100.0},
    }
    return Scenario(
        scenario_id="unit",
        tasks=[Task(task_id=0, work=20.0), Task(task_id=1, work=10.0)],
        edges=[Edge(parent=0, child=1, data=15.0)],
        resources=resources,
        bandwidth=bandwidth,
    )


class SchedulingEnvCorrectnessTest(unittest.TestCase):
    def test_ready_tasks_respect_dependencies(self):
        env = SchedulingEnv(make_two_task_scenario())
        self.assertEqual(env.ready_tasks(), [0])
        self.assertIn((0, "c0"), env.legal_actions())
        self.assertNotIn((1, "c0"), env.legal_actions())

    def test_illegal_action_is_rejected(self):
        env = SchedulingEnv(make_two_task_scenario())
        with self.assertRaises(ValueError):
            env.step((1, "c0"))

    def test_cross_resource_communication_affects_start_time(self):
        env = SchedulingEnv(make_two_task_scenario())
        env.step((0, "c0"))
        estimate = env.estimate(1, "d0")
        self.assertAlmostEqual(estimate["communication_wait"], 3.0)
        self.assertAlmostEqual(estimate["start"], 5.0)
        self.assertAlmostEqual(estimate["finish"], 10.0)

    def test_same_resource_has_no_communication_delay(self):
        env = SchedulingEnv(make_two_task_scenario())
        env.step((0, "c0"))
        estimate = env.estimate(1, "c0")
        self.assertAlmostEqual(estimate["communication_wait"], 0.0)
        self.assertAlmostEqual(estimate["start"], 2.0)
        self.assertAlmostEqual(estimate["finish"], 3.0)

    def test_makespan_is_max_finish_time(self):
        env = SchedulingEnv(make_two_task_scenario())
        env.step((0, "c0"))
        env.step((1, "d0"))
        self.assertTrue(env.done)
        self.assertAlmostEqual(env.makespan, max(env.finish_times.values()))


if __name__ == "__main__":
    unittest.main()
