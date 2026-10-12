"""End-to-end checkpoint, evaluation, generation, and plotting integration."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


class WorkflowCheck(unittest.TestCase):
    def run_script(self, name, *arguments):
        env = {**os.environ, "OMP_NUM_THREADS": "1", "MPLCONFIGDIR": str(self.work / "mpl")}
        result = subprocess.run(
            [sys.executable, "-m", "gpt2." + Path(name).stem, *map(str, arguments)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_resume_evaluation_and_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            self.work = Path(directory)
            data = self.work / "data"
            full, resumed = self.work / "full", self.work / "resumed"
            self.run_script(
                "prepare_data.py",
                "--demo",
                "--tokenizer",
                "byte",
                "--out",
                data,
                "--max-tokens",
                100000,
            )
            common = ["--config", ROOT / "configs/integration.json", "--data", data]
            self.run_script("train.py", *common, "--out", full)
            self.run_script("train.py", *common, "--out", resumed, "--stop-after", 4)
            paused = torch.load(resumed / "last.pt", weights_only=True)
            stop_file = resumed / "stop.request"
            stop_file.write_text('{"reason":"test_interruption"}\n')
            self.run_script("train.py", *common, "--out", resumed, "--resume", resumed / "last.pt")
            stopped = torch.load(resumed / "last.pt", weights_only=True)
            self.assertEqual(stopped["step"], 4)
            for key in paused["model"]:
                self.assertTrue(torch.equal(paused["model"][key], stopped["model"][key]), key)
            stop_file.unlink()
            self.run_script("train.py", *common, "--out", resumed, "--resume", resumed / "last.pt")
            a = torch.load(full / "last.pt", weights_only=True)
            b = torch.load(resumed / "last.pt", weights_only=True)
            self.assertEqual(a["step"], 8)
            self.assertEqual(a["val_loss"], b["val_loss"])
            for key in a["model"]:
                self.assertTrue(torch.equal(a["model"][key], b["model"][key]), key)
            for key, state in a["optimizer"]["state"].items():
                for name, value in state.items():
                    self.assertTrue(torch.equal(value, b["optimizer"]["state"][key][name]))
            self.assertEqual(len(list((full / "snapshots").glob("*.pt"))), 2)
            meta = json.loads((data / "meta.json").read_text())
            self.run_script(
                "evaluate.py",
                "--checkpoint",
                full / "best.pt",
                "--data",
                data,
                "--out",
                full / "evaluation.json",
                "--device",
                "cpu",
                "--verify-data",
            )
            evaluation = json.loads((full / "evaluation.json").read_text())
            self.assertTrue(evaluation["complete_window_sweep"])
            self.assertEqual(evaluation["scored_tokens"], ((meta["tokens"]["val"] - 1) // 64) * 64)
            self.run_script(
                "sample.py",
                "--checkpoint",
                full / "best.pt",
                "--prompts",
                ROOT / "configs/prompts.json",
                "--out",
                full / "samples.jsonl",
                "--max-new-tokens",
                4,
                "--device",
                "cpu",
            )
            self.assertEqual(len((full / "samples.jsonl").read_text().splitlines()), 5)
            self.run_script("plot_metrics.py", "--run", full)
            for filename in (
                "loss.png",
                "loss.svg",
                "optimization.png",
                "optimization.svg",
                "run_summary.json",
            ):
                self.assertTrue((full / "figures" / filename).is_file())


if __name__ == "__main__":
    unittest.main()
