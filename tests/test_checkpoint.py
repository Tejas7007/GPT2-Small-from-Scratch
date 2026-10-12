"""Persistent replacement failures must retain the last resumable checkpoint."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from gpt2.checkpoint import atomic_save


class CheckpointWriteCheck(unittest.TestCase):
    def test_staged_roundtrip_preserves_tied_tensor_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            tensor = torch.arange(100)
            atomic_save({"weights": tensor, "tied_weights": tensor, "step": 3}, path)
            restored = torch.load(path, weights_only=True)
            self.assertTrue(torch.equal(restored["weights"], tensor))
            self.assertEqual(restored["weights"].data_ptr(), restored["tied_weights"].data_ptr())
            self.assertEqual(restored["step"], 3)
            self.assertFalse(path.with_suffix(".tmp").exists())

    def test_partial_persistent_copy_keeps_previous_checkpoint_and_closes_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            atomic_save({"step": 3}, path)
            opened = []

            def fail_copy(source, destination, length):
                opened.extend([source, destination])
                destination.write(source.read(128))
                raise OSError("Persistent volume write failed")

            with patch("gpt2.checkpoint.shutil.copyfileobj", side_effect=fail_copy):
                with self.assertRaisesRegex(OSError, "Persistent volume write failed"):
                    atomic_save({"step": 4}, path)
            self.assertEqual(torch.load(path, weights_only=True)["step"], 3)
            self.assertFalse(path.with_suffix(".tmp").exists())
            self.assertTrue(all(handle.closed for handle in opened))


if __name__ == "__main__":
    unittest.main()
