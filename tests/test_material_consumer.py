import unittest

from env_factory.material_consumer import (
    DATASET_SPLITS,
    assign_dataset_splits,
    consumer_contract,
)


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

    def test_consumer_schema_makes_split_part_of_every_record(self):
        contract = consumer_contract()
        records = contract["records"]
        self.assertEqual(contract["bundle_version"], "6.0")
        self.assertEqual(records["split_unit"], "item_id")
        self.assertIn("split", records["required_fields"])
        self.assertEqual(
            records["json_schema"]["properties"]["split"]["enum"],
            list(DATASET_SPLITS),
        )


if __name__ == "__main__":
    unittest.main()
