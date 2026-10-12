"""Tokenizer adapters with explicit vocabulary and document-boundary IDs.

The GPT-2 BPE vocabulary is used for pretraining. The byte tokenizer is limited
to offline integration tests and is incompatible with the GPT-2 vocabulary.
"""


class Tokenizer:
    """Encode/decode text while exposing the vocabulary size and EOT token."""

    def __init__(self, name="gpt2"):
        self.name = name
        if name == "gpt2":
            import tiktoken

            self.encoding = tiktoken.get_encoding("gpt2")
            self.vocab_size = self.encoding.n_vocab
            self.eot_id = self.encoding.eot_token
        elif name == "byte":
            self.vocab_size = 257
            self.eot_id = 256  # separate document boundary, outside byte IDs 0..255
        else:
            raise ValueError("Tokenizer must be gpt2 or byte.")

    def encode(self, text):
        if self.name == "gpt2":
            # Treat literal special-token-looking text as ordinary corpus text.
            return self.encoding.encode_ordinary(text)
        return list(text.encode("utf-8"))

    def decode(self, ids):
        if self.name == "gpt2":
            return self.encoding.decode(ids)
        return bytes(i for i in ids if i != self.eot_id).decode("utf-8", errors="replace")

    def encode_batch(self, texts, workers=8):
        """Tokenize a bounded document batch while preserving input order."""
        if self.name == "gpt2":
            return self.encoding.encode_ordinary_batch(texts, num_threads=workers)
        return [self.encode(text) for text in texts]
