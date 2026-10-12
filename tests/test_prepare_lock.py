"""Concurrent preparation writers reuse committed state without truncation."""

import hashlib
import json
import os
import select
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_RUNNER = r"""
import os
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, sys.argv.pop(1))
from gpt2 import prepare_data

hub = types.ModuleType("huggingface_hub")
hub.HfApi = lambda: types.SimpleNamespace(
    dataset_info=lambda *args, **kwargs: types.SimpleNamespace(sha="fixture-immutable-revision")
)
sys.modules["huggingface_hub"] = hub

def fixture_documents(args):
    args.consumed_shards.append("fixture/000.parquet")
    for index in range(100):
        yield f"Document {index}: " + "deterministic fixture text " * (index % 5 + 1)

prepare_data.documents = fixture_documents
original_tokenizer = prepare_data.Tokenizer

def fixture_tokenizer(name):
    if os.environ.get("FIXTURE_LOCK_MARKER"):
        Path(os.environ["FIXTURE_LOCK_MARKER"]).touch()
        release = Path(os.environ["FIXTURE_RELEASE"])
        deadline = time.monotonic() + 20
        while not release.exists():
            if time.monotonic() > deadline:
                raise TimeoutError("Fixture writer was not released")
            time.sleep(0.01)
    return original_tokenizer(name)

prepare_data.Tokenizer = fixture_tokenizer
prepare_data.main()
"""


class PreparationLockCheck(unittest.TestCase):
    def command(self, runner, output, *extra):
        return [
            sys.executable,
            str(runner),
            str(ROOT),
            "--hf-dataset",
            "fixture/fineweb",
            "--tokenizer",
            "byte",
            "--val-fraction",
            "0.2",
            "--max-tokens",
            "1000000",
            "--out",
            str(output),
            *extra,
        ]

    def test_waiting_writer_reuses_completed_or_paused_run(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            runner = work / "fixture_runner.py"
            runner.write_text(FIXTURE_RUNNER)
            golden = work / "golden"
            baseline = subprocess.run(
                self.command(runner, golden), capture_output=True, text=True, timeout=30
            )
            self.assertEqual(baseline.returncode, 0, baseline.stderr)
            for paused in (False, True):
                with self.subTest(paused=paused):
                    output = work / f"concurrent-{paused}"
                    marker, release = work / f"marker-{paused}", work / f"release-{paused}"
                    env = {
                        **os.environ,
                        "FIXTURE_LOCK_MARKER": str(marker),
                        "FIXTURE_RELEASE": str(release),
                    }
                    extra = ("--stop-after-documents", "25") if paused else ()
                    first = subprocess.Popen(
                        self.command(runner, output, *extra),
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    second = None
                    try:
                        deadline = time.monotonic() + 10
                        while not marker.exists() and time.monotonic() < deadline:
                            time.sleep(0.01)
                        self.assertTrue(marker.exists(), "First writer failed to acquire its lock")
                        self.assertFalse((output / "preparation.json").exists())
                        second = subprocess.Popen(
                            self.command(runner, output),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                        )
                        ready, _, _ = select.select([second.stdout], [], [], 10)
                        self.assertTrue(ready, "Second writer did not report waiting for the lock")
                        self.assertIn("Waiting for", second.stdout.readline())
                        release.touch()
                        first_out, first_err = first.communicate(timeout=30)
                        second_out, second_err = second.communicate(timeout=30)
                        self.assertEqual(
                            first.returncode, 75 if paused else 0, first_out + first_err
                        )
                        self.assertEqual(second.returncode, 0, second_out + second_err)
                    finally:
                        release.touch()
                        for process in (first, second):
                            if process is not None and process.poll() is None:
                                process.kill()
                                process.communicate()
                    meta = json.loads((output / "meta.json").read_text())
                    self.assertEqual(meta, json.loads((golden / "meta.json").read_text()))
                    for split in ("train", "val"):
                        actual = (output / f"{split}.bin").read_bytes()
                        self.assertEqual(actual, (golden / f"{split}.bin").read_bytes())
                        self.assertEqual(hashlib.sha256(actual).hexdigest(), meta["sha256"][split])
                    # A fresh invocation must still explicitly opt into reuse.
                    repeated = subprocess.run(
                        self.command(runner, output), capture_output=True, text=True, timeout=30
                    )
                    self.assertEqual(repeated.returncode, 2)
                    self.assertIn("Output already has data files", repeated.stderr)


if __name__ == "__main__":
    unittest.main()
