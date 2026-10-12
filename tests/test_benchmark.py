"""Training runtime projections include overhead and exclude update warmup."""

import unittest

from gpt2.benchmark import project_run


class BenchmarkForecastChecks(unittest.TestCase):
    def manifest(self):
        return {
            "total_steps": 100,
            "tokens_per_step": 524288,
            "args": {"eval_interval": 25, "milestone_tokens": [5242880, 13107200]},
        }

    def records(self):
        return [
            {"kind": "train", "step": 1, "seconds": 1000},
            {"kind": "train", "step": 2, "seconds": 1},
            {"kind": "train", "step": 3, "seconds": 3},
            {"kind": "val", "validation_seconds": 5},
            {"kind": "checkpoint", "checkpoint_seconds": 2},
        ]

    def test_mean_time_excludes_warmup_and_includes_unique_evaluations(self):
        report = project_run(self.records(), self.manifest(), 1)
        # 200 seconds of training; initial + 5 unique validation points;
        # 5 checkpoints take 10 seconds.
        self.assertEqual(report["measured_updates"], 2)
        self.assertEqual(report["global_tokens_per_second"], 262144)
        self.assertAlmostEqual(report["projected_gpu_session_hours"], 240 / 3600)

    def test_final_evaluation_is_included_between_regular_intervals(self):
        manifest = self.manifest()
        manifest["total_steps"] = 99
        report = project_run(self.records(), manifest, 1)
        self.assertAlmostEqual(report["projected_gpu_session_hours"], 238 / 3600)

    def test_missing_or_invalid_measurements_are_rejected(self):
        for records in ([], self.records()[:-1], self.records()[1:3]):
            with self.assertRaises(ValueError):
                project_run(records, self.manifest(), 1)
        records = self.records()
        records[2]["seconds"] = float("nan")
        with self.assertRaises(ValueError):
            project_run(records, self.manifest(), 1)


if __name__ == "__main__":
    unittest.main()
