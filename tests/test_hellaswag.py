"""Check ending-only likelihood alignment independently of dataset downloads."""

import unittest

import torch

from gpt2.evaluate_hellaswag import encode_choices, ending_nll, make_batch
from gpt2.tokenizer import Tokenizer


class HellaSwagScoringTests(unittest.TestCase):
    def test_first_ending_token_and_padding_mask(self):
        x, y, mask = make_batch([([1, 2, 3, 4], 2), ([1, 2, 5], 2)], 0, "cpu")
        self.assertEqual(x.tolist(), [[1, 2, 3], [1, 2, 0]])
        self.assertEqual(y.tolist(), [[2, 3, 4], [2, 5, 0]])
        self.assertEqual(mask.tolist(), [[False, True, True], [False, True, False]])

    def test_prompt_and_padding_cannot_change_scores(self):
        _, y, mask = make_batch([([1, 2, 3, 4], 2), ([1, 2, 5], 2)], 0, "cpu")
        logits = torch.zeros(2, 3, 6)
        logits[0, 1, 3] = 1.5
        logits[0, 2, 4] = 1.5
        logits[1, 1, 5] = 1
        sums, counts = ending_nll(logits, y, mask)
        changed = logits.clone()
        changed[~mask] = 1000 * torch.randn_like(changed[~mask])
        self.assertTrue(torch.equal(sums, ending_nll(changed, y, mask)[0]))
        self.assertEqual(counts.tolist(), [2, 1])
        # The longer ending has a larger total NLL but lower per-token NLL.
        self.assertEqual(sums.argmin().item(), 1)
        self.assertEqual((sums / counts).argmin().item(), 0)

    def test_encoding_does_not_read_gold_labels(self):
        row = {
            "activity_label": "Test",
            "ctx_a": "A person runs.",
            "ctx_b": "then",
            "endings": ["walks.", "jumps.", "stops.", "sits."],
        }
        tokenizer = Tokenizer("byte")
        without = encode_choices(row, tokenizer, 1024)
        row["label"] = "3"
        self.assertEqual(without, encode_choices(row, tokenizer, 1024))
        with self.assertRaises(ValueError):
            encode_choices(row, tokenizer, 4)


if __name__ == "__main__":
    unittest.main()
