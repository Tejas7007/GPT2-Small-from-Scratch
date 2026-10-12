"""Decoder-only transformer with the GPT-2 Small architecture.

Tensor dimensions follow B (batch), T (sequence), C (embedding width), H
(attention heads), D (head width), and V (vocabulary). Each prediction at
position t depends only on input positions <= t. Parameters are initialized
randomly; the module does not load pretrained model weights.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    """Architecture and attention backend, serialized alongside model weights."""

    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    attention: str = "sdpa"

    def __post_init__(self):
        if min(self.vocab_size, self.block_size, self.n_layer, self.n_head, self.n_embd) < 1:
            raise ValueError("All model dimensions must be positive.")
        if self.n_embd % self.n_head:
            raise ValueError("Embedding width must divide evenly into heads.")
        if self.attention not in {"manual", "sdpa"}:
            raise ValueError("attention must be manual or sdpa.")


class CausalSelfAttention(nn.Module):
    """Multi-head causal attention with reference and optimized implementations."""

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.attention = config.attention
        # A fused projection computes Q, K, and V in a single matrix multiply.
        # Splitting its output preserves three independent learned projections.
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)  # three [B, T, C] tensors
        # [B,T,C] -> [B,H,T,D]. The final two axes are the matrix dimensions
        # used by attention; batch and head axes broadcast independently.
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.attention == "manual":
            # QK^T measures compatibility between query/key projections. Scale
            # by sqrt(D) to control score variance as the head width increases.
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # scores: [B,H,T,T]. Mask j > i before softmax so future input
            # positions receive zero attention mass. The diagonal stays visible:
            # output i predicts target i+1 rather than reconstructing input i.
            future = torch.ones(T, T, device=x.device, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(future, float("-inf"))
            weights = F.softmax(scores, dim=-1)
            y = weights @ v  # weighted sum of values: [B, H, T, D]
        else:
            # SDPA selects an available backend, including fused CUDA kernels
            # on compatible devices. No explicit T-by-T mask is stored here.
            # The reference branch above supports numerical parity checks.
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)  # learned mixing of concatenated head outputs


class FeedForward(nn.Module):
    """Position-wise expansion/projection with the GPT-2 GELU approximation."""

    def __init__(self, config):
        super().__init__()
        self.expand = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        # Unlike attention, the feedforward sublayer does not mix sequence
        # positions. Its weights are shared across positions and batch elements.
        return self.proj(F.gelu(self.expand(x), approximate="tanh"))


class TransformerBlock(nn.Module):
    """Pre-normalized attention and feedforward sublayers with residual paths."""

    def __init__(self, config):
        super().__init__()
        self.ln_attention = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.attention = CausalSelfAttention(config)
        self.ln_ff = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.ff = FeedForward(config)

    def forward(self, x):
        # Normalize within the feature axis before each residual update. The
        # identity paths preserve a direct route for activations and gradients
        # across depth while the sublayers learn additive transformations.
        x = x + self.attention(self.ln_attention(x))
        x = x + self.ff(self.ln_ff(x))
        return x


class GPT(nn.Module):
    """Causal language model with learned positions and tied vocabulary weights."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        # Each vocabulary ID indexes a trainable feature vector. Position
        # embeddings encode offsets within the current context window.
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_final = nn.LayerNorm(config.n_embd, eps=1e-5)
        # Share the [V,C] matrix between input lookup and output classification.
        # Optimizer parameter enumeration must include this parameter only once.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        self.lm_head.weight = self.token_embedding.weight
        # Scale the two residual-output projections per block by sqrt(2L).
        # This controls accumulated residual variance at initialization.
        for block in self.blocks:
            for projection in (block.attention.proj, block.ff.proj):
                nn.init.normal_(projection.weight, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, token_ids, targets=None):
        """Return logits and optional mean next-token cross-entropy.

        token_ids and targets have shape [B,T]; targets must already be shifted
        one position ahead. Without targets, only the final position's logits
        are computed to avoid projecting unused positions during generation.
        """
        if token_ids.ndim != 2:
            raise ValueError("Input must have shape [B, T].")
        B, T = token_ids.shape
        if not 1 <= T <= self.config.block_size:
            raise ValueError("Input length must fit the configured context window.")
        positions = torch.arange(T, device=token_ids.device)
        # Position vectors broadcast across the batch: [B,T,C] + [T,C].
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        if targets is None:
            # Generation only needs the last position's next-token prediction.
            return self.lm_head(x[:, -1:, :]), None
        if targets.shape != token_ids.shape:
            raise ValueError("Targets must have the same shape as input.")
        logits = self.lm_head(x)  # [B, T, V], unnormalized scores
        # Flatten B/T into independent classification observations. PyTorch's
        # cross-entropy applies log-softmax internally to unnormalized logits.
        loss = F.cross_entropy(logits.reshape(B * T, -1), targets.reshape(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, token_ids, max_new_tokens=100, temperature=0.8, top_k=50):
        """Sample autoregressively with temperature and optional top-k filtering.

        Context is recomputed for each token. This implementation prioritizes
        reproducible sampling over serving throughput and does not use a KV cache.
        """
        if temperature <= 0 or max_new_tokens < 0 or (top_k is not None and top_k < 1):
            raise ValueError("Use positive temperature/top_k and nonnegative token count.")
        for _ in range(max_new_tokens):
            # Truncate context to the learned position table's maximum length.
            context = token_ids[:, -self.config.block_size :]
            logits, _ = self(context)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                cutoff = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))
            next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            token_ids = torch.cat((token_ids, next_id), dim=1)
        return token_ids

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
