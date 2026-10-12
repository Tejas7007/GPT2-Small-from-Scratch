"""Optional accuracy figures require the complete, identical scoring protocol."""

import json
import tempfile
import unittest
from pathlib import Path

from gpt2.evaluate_hellaswag import DATASET, DATASET_REVISION, EVALUATION_PROTOCOL, VALIDATION_ROWS
from gpt2.plot_metrics import matching_hellaswag_reports, plot_hellaswag_quality


class QualityPlotCheck(unittest.TestCase):
    def test_matching_fixture_draws_and_incompatible_results_are_omitted(self):
        common = {
            "task": "hellaswag",
            "split": "validation",
            "fewshot": 0,
            "dataset": DATASET,
            "dataset_revision": DATASET_REVISION,
            "evaluated_examples": VALIDATION_ROWS,
            "available_examples": VALIDATION_ROWS,
            "complete_validation_sweep": True,
            "validation_rows_sha256": "a" * 64,
            "tokenizer": "gpt2",
            "context_length": 1024,
            "chance_accuracy": 0.25,
            "precision": "bfloat16",
            "evaluation_protocol": EVALUATION_PROTOCOL,
            "accuracy": 0.26,
            "accuracy_token_normalized": 0.29,
            "accuracy_character_normalized": 0.28,
        }
        model = {**common, "checkpoint": "best.pt"}
        reference = {**common, "checkpoint": "openai-community/gpt2", "accuracy": 0.28}
        self.assertTrue(matching_hellaswag_reports(model, reference))
        for difference in (
            {"validation_rows_sha256": "b" * 64},
            {"evaluated_examples": 100},
            {"complete_validation_sweep": False},
            {"evaluation_protocol": "different scoring"},
            {"precision": "float32"},
        ):
            self.assertFalse(matching_hellaswag_reports(model, {**reference, **difference}))
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            out = run / "figures"
            out.mkdir()
            (run / "hellaswag.json").write_text(json.dumps(model))
            (run / "hellaswag_reference.json").write_text(json.dumps(reference))
            self.assertTrue(plot_hellaswag_quality(run, out))
            for filename in ("quality.png", "quality.svg"):
                self.assertTrue((out / filename).is_file())
            (run / "hellaswag_status.json").write_text('{"status": "failed_or_timed_out"}')
            self.assertFalse(plot_hellaswag_quality(run, out))


if __name__ == "__main__":
    unittest.main()
