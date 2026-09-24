import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from examples.generate_task import _existing_task_numbers, _reserve_task_directories
from env_factory.task_routing import (
    allocate_training_routes,
    compatible_training_categories,
    parse_training_mix,
    select_training_intent,
)


class IncrementalTaskDirectoryTest(unittest.TestCase):
    def test_noise_resistance_is_not_an_active_training_category(self):
        with self.assertRaisesRegex(ValueError, "invalid training mix entry"):
            parse_training_mix(
                "direct_response=.2,simple_agentic=.3,multi_step_agentic=.4,noise_resistance=.1"
            )

    def test_route_intent_mapping_keeps_explanations_out_of_agentic_routes(self):
        self.assertEqual(compatible_training_categories("explain"), ("direct_response",))
        self.assertEqual(select_training_intent("direct_response", "explain"), "explain")
        with self.assertRaisesRegex(ValueError, "incompatible"):
            select_training_intent("multi_step_agentic", "explain")

    def test_explicit_intent_allocates_only_compatible_routes(self):
        routes = allocate_training_routes(
            12,
            parse_training_mix(None),
            allowed_categories=compatible_training_categories("recommend"),
        )
        self.assertEqual(set(routes), {"simple_agentic"})

    def test_default_training_mix_allocates_expected_twenty_task_quota(self):
        routes = allocate_training_routes(20, parse_training_mix(None))
        self.assertEqual(routes.count("direct_response"), 4)
        self.assertEqual(routes.count("simple_agentic"), 6)
        self.assertEqual(routes.count("multi_step_agentic"), 10)

    def test_five_task_batch_preserves_every_curriculum_category(self):
        routes = allocate_training_routes(5, parse_training_mix(None))
        self.assertEqual(routes.count("direct_response"), 1)
        self.assertEqual(routes.count("simple_agentic"), 2)
        self.assertEqual(routes.count("multi_step_agentic"), 2)
        self.assertEqual(set(routes), {
            "direct_response", "simple_agentic", "multi_step_agentic",
        })

    def test_reservation_appends_after_highest_existing_task(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "task-1").mkdir()
            (root / "task-3").mkdir()
            (root / "task-draft").mkdir()

            reserved = _reserve_task_directories(root, 2)

            self.assertEqual([number for number, _ in reserved], [4, 5])
            self.assertEqual(_existing_task_numbers(root), [1, 3, 4, 5])
            self.assertTrue((root / "task-1").is_dir())
            self.assertTrue((root / "task-3").is_dir())

    def test_sequential_runs_never_reuse_previous_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            first = _reserve_task_directories(root, 1)
            second = _reserve_task_directories(root, 2)

            self.assertEqual([number for number, _ in first], [1])
            self.assertEqual([number for number, _ in second], [2, 3])

    def test_concurrent_reservations_do_not_collide(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with ThreadPoolExecutor(max_workers=2) as executor:
                batches = list(executor.map(
                    lambda _: _reserve_task_directories(root, 3),
                    range(2),
                ))

            numbers = [number for batch in batches for number, _ in batch]
            self.assertEqual(len(numbers), 6)
            self.assertEqual(len(set(numbers)), 6)


if __name__ == "__main__":
    unittest.main()
