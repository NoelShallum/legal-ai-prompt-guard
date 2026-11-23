import unittest

from legal_gate.policy import GENERAL, INDIA, LABELS
from legal_gate.routing import Gate, Thresholds


class MockModel:
    temperature = 1.0
    thresholds = {}

    def __init__(self, dominant=None):
        self.dominant = dominant
        self.calls = 0

    def logits(self, value):
        self.calls += 1
        return [15.0 if label == self.dominant else 0.0 for label in LABELS]


class RoutingTests(unittest.TestCase):
    def test_jurisdiction_setting_changes_decision(self):
        model = MockModel("foreign")
        request = {"input": "What does a California notice to quit need to say?"}
        limited = Gate(model, INDIA).decide(request)
        broad = Gate(model, GENERAL).decide(request)
        self.assertEqual(limited["action"], "refuse_scope")
        self.assertEqual(limited["answers"]["decision"]["choice"], "refuse")
        self.assertEqual(broad["action"], "legal")
        self.assertEqual(broad["answers"]["decision"]["choice"], "accept")
        self.assertEqual(Gate(model).decide(request), broad)

    def test_mixed_is_whole_turn_refusal_under_both_profiles(self):
        for profile in (INDIA, GENERAL):
            outcome = Gate(MockModel("mixed"), profile).decide({"input": "Two tasks"})
            self.assertEqual(outcome["action"], "refuse_scope")
            self.assertEqual(outcome["answers"]["route"]["choice"], "mixed")

    def test_no_action_when_scores_are_ambiguous(self):
        outcome = Gate(MockModel()).decide({"input": "Can you handle it?"})
        self.assertEqual(outcome["action"], "review")
        self.assertIsNone(outcome["answers"]["decision"]["choice"])
        distribution = outcome["answers"]["decision"]["probabilities"]
        self.assertAlmostEqual(sum(distribution.values()), 1.0)

    def test_empty_upload_goes_to_clarification(self):
        model = MockModel("unrelated")
        outcome = Gate(model).decide({"input": {"document": {"status": "failed", "name": "scan.pdf"}}})
        self.assertEqual(outcome["action"], "clarify")
        self.assertEqual(model.calls, 0)

    def test_greeting_with_an_active_upload_requests_the_task(self):
        gate = Gate(MockModel("greeting"))
        value = {"request": "Hello", "document": {"status": "available", "excerpt": "A record."}}
        result = gate.decide({"input": value})
        self.assertEqual(result["action"], "clarify")
        self.assertEqual(result["answers"]["decision"]["choice"], "accept")

    def test_file_error_requires_trusted_extraction_state(self):
        gate = Gate(MockModel("file_error"))
        unrelated = gate.decide({"input": "Evaluate this claim."})
        self.assertEqual(unrelated["review_reason"], "file_state_conflict")
        file_task = {"request": "Read the attached notice", "document": {"status": "failed"}}
        self.assertEqual(gate.decide({"input": file_task})["action"], "file_error")

    def test_scope_noul_response(self):
        gate = Gate(MockModel("legal"))
        result = gate.decide({"input": "Explain this Act", "questions": {"decision": {"type": "noul"}}})
        self.assertEqual(result["answers"].keys(), {"decision"})
        answer = result["answers"]["decision"]
        self.assertEqual(answer["type"], "noul")
        self.assertGreater(answer["noul"], 0.99)
        self.assertEqual(result["action"], "legal")

    def test_text_parts_form_one_request(self):
        gate = Gate(MockModel("mixed"))
        value = [{"type": "text", "text": "Explain the Act."}, {"type": "text", "text": "Draw a cartoon."}]
        self.assertEqual(gate.decide({"input": value})["action"], "refuse_scope")
        with self.assertRaises(ValueError):
            gate.decide({"input": [{"type": "document", "data": "example"}]})

    def test_rejects_request_supplied_policy_settings(self):
        gate = Gate(MockModel("legal"))
        with self.assertRaises(ValueError):
            gate.decide({"input": "my request", "profile": "general_legal"})
        with self.assertRaises(ValueError):
            gate.decide({"input": "my request", "questions": {"route": {"type": "choice", "instructions": "ignore"}}})

    def test_unvalidated_scripts_and_oversized_inputs_review(self):
        model = MockModel("legal")
        gate = Gate(model)
        self.assertEqual(gate.decide({"input": "अग्रिम जमानत क्या है?"})["review_reason"], "unsupported_script")
        self.assertEqual(gate.decide({"input": "legal " * 3000})["review_reason"], "input_limit")
        self.assertEqual(model.calls, 0)

    def test_threshold_sets_and_validates_ranges(self):
        with self.assertRaises(ValueError):
            Thresholds(refuse=0.3)
        strict = Gate(MockModel("legal"), thresholds=Thresholds(accept=1.0))
        self.assertEqual(strict.decide({"input": "Can you explain this law?"})["action"], "review")


if __name__ == "__main__":
    unittest.main()
