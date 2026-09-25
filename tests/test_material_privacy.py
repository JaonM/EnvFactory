import unittest

from env_factory.material_privacy import audit_rollout_privacy
from env_factory.trajectory_schema import policy_transition


class MaterialPrivacyTest(unittest.TestCase):
    @staticmethod
    def rollout(result=None):
        return {
            "schema_version": "2.0",
            "material_visibility_version": "1.0",
            "episodes": [{
                "transitions": [{
                    "step": 0,
                    "agent_input": [{"role": "user", "content": "help"}],
                    "assistant_output": '{"kind":"respond","content":"done"}',
                    "observation": {},
                    "action": {"kind": "respond", "content": "done"},
                    "result": result or {"status": 200, "user_query": "thanks"},
                    "next_observation": {},
                    "reward": 1.0,
                    "terminated": True,
                    "truncated": False,
                    "trainer_metadata": {"user_simulator": {
                        "outcome_category": "user_acceptance",
                        "transition_id": "accept",
                    }},
                }],
            }],
        }

    def test_trainer_metadata_is_not_scanned_as_policy_visible(self):
        report = audit_rollout_privacy(self.rollout())
        self.assertTrue(report["eligible_for_policy_training_export"])
        projected = policy_transition(
            self.rollout()["episodes"][0]["transitions"][0]
        )
        self.assertNotIn("trainer_metadata", projected)

    def test_credential_in_visible_tool_result_is_rejected_without_echoing_it(self):
        secret = "sk-abcdefghijklmnopqrstuvwxyz123456"
        report = audit_rollout_privacy(self.rollout({"status": 200, "token": secret}))
        self.assertFalse(report["eligible_for_policy_training_export"])
        self.assertTrue(report["credential_findings"])
        self.assertNotIn(secret, str(report))

    def test_pii_generated_during_rollout_is_not_exportable(self):
        report = audit_rollout_privacy(self.rollout({
            "status": 200,
            "user_query": "email me at fixture@example.test",
        }))
        self.assertFalse(report["eligible_for_policy_training_export"])
        self.assertEqual(report["pii_findings"][0]["kind"], "email")

    def test_hidden_control_key_in_visible_observation_is_rejected(self):
        rollout = self.rollout()
        rollout["episodes"][0]["transitions"][0]["observation"] = {
            "acceptance_contract": {"answer": "hidden"},
        }
        report = audit_rollout_privacy(rollout)
        self.assertFalse(report["eligible_for_policy_training_export"])
        self.assertEqual(
            report["forbidden_key_findings"][0]["kind"], "acceptance_contract"
        )

    def test_visibility_marker_is_mandatory(self):
        rollout = self.rollout()
        rollout.pop("material_visibility_version")
        self.assertFalse(
            audit_rollout_privacy(rollout)["eligible_for_policy_training_export"]
        )


if __name__ == "__main__":
    unittest.main()
