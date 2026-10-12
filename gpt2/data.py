"""Memory-mapped token batches with deterministic, rank-disjoint epoch sampling."""

from pathlib import Path

import numpy as np
import torch


class TokenStream:
    """Produce causal input/target windows from a little-endian uint16 stream.

    Shuffled mode partitions the corpus into non-overlapping T-token input
    windows. All ranks reconstruct one shared permutation; each global batch
    assigns distinct windows to ranks. Incomplete global batches are dropped
    at epoch boundaries. The one-token target shift crosses window boundaries.

    Random mode samples windows with replacement for isolated behavior checks.
    Both modes expose serializable sampling state for checkpoint continuation.
    """

    def __init__(
        self, path, block_size, batch_size, seed, device, mode="shuffled", rank=0, world_size=1
    ):
        if block_size < 1 or batch_size < 1 or not 0 <= rank < world_size:
            raise ValueError("Invalid sequence length, batch size, or rank configuration.")
        if mode not in {"shuffled", "random"}:
            raise ValueError("Sampling mode must be shuffled or random.")
        self.tokens = np.memmap(Path(path), dtype="<u2", mode="r")
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.seed, self.mode = seed, mode
        self.rank, self.world_size = rank, world_size
        self.generator = torch.Generator().manual_seed(seed + rank)
        self.epoch, self.cursor = 0, 0
        self.window_count = (len(self.tokens) - 1) // block_size
        self.global_batch = batch_size * world_size
        if len(self.tokens) <= block_size:
            raise ValueError(f"{path} needs more than {block_size} tokens.")
        if mode == "shuffled" and self.window_count < self.global_batch:
            raise ValueError("Corpus must contain at least one complete global batch.")
        self.permutation = None
        if mode == "shuffled":
            self._reset_permutation()

    def _reset_permutation(self):
        # Reconstruct the permutation from seed/epoch instead of checkpointing
        # its entire index array. Every rank uses the same CPU permutation.
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.permutation = torch.randperm(self.window_count, generator=generator)

    def state_dict(self):
        """Return the cursor/RNG and batch-layout identity needed for continuation.

        The potentially large permutation is reconstructed from seed and epoch.
        A rank-local random generator is retained for replacement sampling.
        """
        return {
            "mode": self.mode,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "generator": self.generator.get_state(),
            "seed": self.seed,
            "rank": self.rank,
            "world_size": self.world_size,
            "block_size": self.block_size,
            "batch_size": self.batch_size,
        }

    def load_state_dict(self, state):
        """Restore only a sampler with the same rank, seed, and batch geometry."""
        for key in ("mode", "seed", "rank", "world_size", "block_size", "batch_size"):
            if state[key] != getattr(self, key):
                raise ValueError(f"Sampler configuration differs at {key}.")
        self.epoch, self.cursor = state["epoch"], state["cursor"]
        self.generator.set_state(state["generator"])
        if self.mode == "shuffled":
            self._reset_permutation()

    def batch(self):
        """Return int64 input/target tensors of shape [batch_size, block_size].

        Corpus storage is uint16, which fits GPT-2's 50,257-token vocabulary.
        Embedding lookup requires integer indices, so only the selected windows
        are copied into int64 tensors rather than expanding the entire corpus.
        """
        if self.mode == "random":
            starts = torch.randint(
                len(self.tokens) - self.block_size, (self.batch_size,), generator=self.generator
            ).tolist()
        else:
            if self.cursor + self.global_batch > self.window_count:
                self.epoch += 1
                self.cursor = 0
                self._reset_permutation()
            begin = self.cursor + self.rank * self.batch_size
            indices = self.permutation[begin : begin + self.batch_size]
            starts = (indices * self.block_size).tolist()
            self.cursor += self.global_batch
        chunks = np.stack([self.tokens[s : s + self.block_size + 1] for s in starts])
        chunks = torch.from_numpy(chunks.astype(np.int64))
        # Equal-length windows permit exact mean-loss gradient accumulation.
        # x[:,t] predicts y[:,t], the following token in the source stream.
        x, y = chunks[:, :-1].contiguous(), chunks[:, 1:].contiguous()
        if self.device.type == "cuda":
            x, y = x.pin_memory(), y.pin_memory()
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
