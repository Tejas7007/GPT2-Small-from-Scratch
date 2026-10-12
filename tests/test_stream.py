"""Sampler invariants without a distributed transport dependency."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from gpt2.data import TokenStream


class StreamChecks(unittest.TestCase):
    def test_rank_windows_are_disjoint_and_cover_complete_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.bin"
            np.arange(257, dtype="<u2").tofile(path)
            ranks = [
                TokenStream(path, 8, 2, 19, "cpu", rank=rank, world_size=2) for rank in range(2)
            ]
            observed = []
            for _ in range(8):
                for stream in ranks:
                    x, y = stream.batch()
                    observed.extend(x[:, 0].tolist())
                    self.assertTrue(torch.equal(x[:, 1:], y[:, :-1]))
            self.assertEqual(len(observed), len(set(observed)))
            self.assertEqual(set(observed), set(range(0, 256, 8)))
            for stream in ranks:
                stream.batch()
                self.assertEqual(stream.epoch, 1)

    def test_sampler_state_restores_across_epoch_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.bin"
            np.arange(97, dtype="<u2").tofile(path)
            original = TokenStream(path, 8, 2, 23, "cpu")
            for _ in range(5):
                original.batch()
            state = original.state_dict()
            restored = TokenStream(path, 8, 2, 23, "cpu")
            restored.load_state_dict(state)
            for _ in range(4):
                a, b = original.batch(), restored.batch()
                self.assertTrue(torch.equal(a[0], b[0]))
                self.assertTrue(torch.equal(a[1], b[1]))
                self.assertEqual(original.epoch, restored.epoch)


if __name__ == "__main__":
    unittest.main()
