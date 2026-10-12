"""Numerical and behavioral invariants for the causal model implementation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from data import TokenStream
from model import GPT, GPTConfig


class ModelChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        self.config = GPTConfig(
            vocab_size=16, block_size=16, n_layer=2, n_head=2, n_embd=32, attention="manual"
        )

    def test_gpt2_parameter_count_and_tying(self):
        # Meta allocation verifies the full architecture's parameter count
        # without materializing its weights in host or device memory.
        with torch.device("meta"):
            full = GPT(GPTConfig())
        self.assertEqual(full.parameter_count(), 124_439_808)
        self.assertIs(full.token_embedding.weight, full.lm_head.weight)

    def test_future_tokens_cannot_change_past_predictions(self):
        for attention in ("manual", "sdpa"):
            config = GPTConfig(**{**vars(self.config), "attention": attention})
            model = GPT(config).eval()
            x = torch.randint(16, (2, 8))
            changed = x.clone()
            changed[:, 4:] = (changed[:, 4:] + 1) % 16
            with torch.no_grad():
                original, _ = model(x, x)
                altered, _ = model(changed, x)
            torch.testing.assert_close(original[:, :4], altered[:, :4], atol=1e-6, rtol=1e-5)

    def test_manual_and_optimized_attention_match_outputs_and_gradients(self):
        manual = GPT(self.config)
        optimized = GPT(GPTConfig(**{**vars(self.config), "attention": "sdpa"}))
        optimized.load_state_dict(manual.state_dict())
        x = torch.randint(16, (2, 8))
        y = torch.randint(16, (2, 8))
        a, loss_a = manual(x, y)
        b, loss_b = optimized(x, y)
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
        loss_a.backward()
        loss_b.backward()
        for p, q in zip(manual.parameters(), optimized.parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-4)

    def test_target_shift(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.bin"
            np.arange(100, dtype="<u2").tofile(path)
            stream = TokenStream(path, 8, 2, 1, torch.device("cpu"))
            x, y = stream.batch()
            self.assertTrue(torch.equal(x[:, 1:], y[:, :-1]))
            self.assertTrue(torch.equal(x + 1, y))

    def test_accumulated_gradients_match_large_batch(self):
        whole = GPT(self.config)
        accumulated = GPT(self.config)
        accumulated.load_state_dict(whole.state_dict())
        x = torch.randint(16, (4, 8))
        y = torch.randint(16, (4, 8))
        whole(x, y)[1].backward()
        for a, b in zip(x.chunk(2), y.chunk(2)):
            (accumulated(a, b)[1] / 2).backward()
        for p, q in zip(whole.parameters(), accumulated.parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-4)

    def test_tiny_batch_can_be_memorized(self):
        model = GPT(self.config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0)
        x = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]]).repeat(2, 1)
        y = (x + 1) % 4
        initial = model(x, y)[1].item()
        for _ in range(120):
            optimizer.zero_grad(set_to_none=True)
            loss = model(x, y)[1]
            loss.backward()
            optimizer.step()
        final = model(x, y)[1].item()
        print(f"\nTiny-batch loss: {initial:.4f} -> {final:.4f}")
        self.assertLess(final, 0.05)
        generated = model.eval().generate(x[:1], max_new_tokens=4, top_k=1)
        self.assertEqual(generated[0, -4:].tolist(), [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main(verbosity=2)
