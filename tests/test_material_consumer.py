import unittest

from env_factory.material_consumer import (
    DATASET_SPLITS,
    assign_dataset_splits,
    consumer_contract,
)
from env_factory.task_similarity import task_family_ids


class MaterialConsumerTest(unittest.TestCase):
    def test_split_assignment_is_deterministic_and_stratified(self):
        items = [
            {"item_id": f"{category}-{index:03d}", "category": category}
            for category in ("direct_response", "simple_agentic", "multi_step_agentic")
            for index in range(20)
        ]
        first = assign_dataset_splits(items)
        second = assign_dataset_splits(list(reversed(items)))
        self.assertEqual(first, second)
        for category in {item["category"] for item in items}:
            counts = {
                split: sum(
                    first[item["item_id"]] == split
                    for item in items if item["category"] == category
                )
                for split in DATASET_SPLITS
            }
            self.assertEqual(counts, {"train": 16, "validation": 2, "test": 2})

    def test_tiny_category_stays_train_only(self):
        assignments = assign_dataset_splits([
            {"item_id": "one", "category": "simple_agentic"},
            {"item_id": "two", "category": "simple_agentic"},
        ])
        self.assertEqual(set(assignments.values()), {"train"})

    def test_near_duplicate_family_is_kept_in_one_split(self):
        family_inputs = [
            {"item_id": "variant-a", "task": "为客户 100 创建订单并核验库存"},
            {"item_id": "variant-b", "task": "为客户 200 创建订单并核验库存"},
            {"item_id": "different", "task": "汇总航班延误并发送报告"},
        ]
        families = task_family_ids(family_inputs)
        self.assertEqual(families["variant-a"], families["variant-b"])
        self.assertNotEqual(families["variant-a"], families["different"])
        items = [
            {
                "item_id": item["item_id"],
                "category": "multi_step_agentic",
                "task_family_id": families[item["item_id"]],
            }
            for item in family_inputs
        ]
        assignments = assign_dataset_splits(items)
        self.assertEqual(assignments["variant-a"], assignments["variant-b"])

    def test_consumer_schema_makes_split_part_of_every_record(self):
        contract = consumer_contract()
        records = contract["records"]
        self.assertEqual(contract["bundle_version"], "9.0")
        self.assertEqual(records["split_unit"], "item_id")
        self.assertIn("split", records["required_fields"])
        self.assertIn("task_family_id", records["required_fields"])
        self.assertIn("generation_model", records["required_fields"])
        self.assertEqual(records["near_duplicate_split_unit"], "task_family_id")
        self.assertEqual(
            records["json_schema"]["properties"]["split"]["enum"],
            list(DATASET_SPLITS),
        )


if __name__ == "__main__":
    unittest.main()
