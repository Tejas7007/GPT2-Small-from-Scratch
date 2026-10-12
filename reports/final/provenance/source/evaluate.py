"""Evaluate a checkpoint on contiguous, non-overlapping held-out token windows."""

import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from model import GPT, GPTConfig
from train import file_sha256


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument(
        "--reference",
        choices=["gpt2"],
        help="Evaluate released GPT-2 weights on the same held-out text",
    )
    parser.add_argument("--reference-revision", default="main")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--max-batches", type=int, help="Limit evaluation; omit for a complete window sweep"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--verify-data", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.max_batches is not None and args.max_batches < 1):
        parser.error("Batch limits must be positive.")
    device_name = args.device
    if device_name == "auto":
        device_name = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    device = torch.device(device_name)
    checkpoint = (
        torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if args.checkpoint
        else None
    )
    meta = json.loads((args.data / "meta.json").read_text())
    if checkpoint is not None and meta != checkpoint["data_meta"]:
        raise ValueError("Evaluation dataset metadata differs from checkpoint provenance.")
    path = args.data / "val.bin"
    if path.stat().st_size != meta["tokens"]["val"] * 2:
        raise ValueError("Validation token file size differs from metadata.")
    if args.verify_data and file_sha256(path) != meta["sha256"]["val"]:
        raise ValueError("Validation token digest differs from metadata.")
    reference_revision = None
    if checkpoint is not None:
        config = GPTConfig(**checkpoint["config"])
        model = GPT(config)
        model.load_state_dict(checkpoint["model"])
    else:
        if meta["tokenizer"] != "gpt2" or meta["vocab_size"] != 50257:
            raise ValueError("Reference evaluation requires GPT-2 tokenization.")
        from huggingface_hub import HfApi
        from transformers import AutoModelForCausalLM

        identifier = "openai-community/gpt2"
        reference_revision = HfApi().model_info(identifier, revision=args.reference_revision).sha
        model = AutoModelForCausalLM.from_pretrained(identifier, revision=reference_revision)
        config = GPTConfig()
    model.to(device).eval()
    tokens = np.memmap(path, dtype="<u2", mode="r")
    window_count = (len(tokens) - 1) // config.block_size
    if window_count < 1:
        raise ValueError("Held-out corpus is shorter than one context window.")
    limit = window_count
    if args.max_batches is not None:
        limit = min(limit, args.max_batches * args.batch_size)
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    def autocast():
        return torch.autocast("cuda", dtype=dtype) if device.type == "cuda" else nullcontext()

    nll_sum, scored_tokens = 0.0, 0
    begin = time.perf_counter()
    for first in range(0, limit, args.batch_size):
        starts = [
            index * config.block_size for index in range(first, min(first + args.batch_size, limit))
        ]
        array = np.stack(
            [tokens[start : start + config.block_size + 1] for start in starts]
        ).astype(np.int64)
        batch = torch.from_numpy(array).to(device)
        x, y = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
        with autocast():
            if checkpoint is not None:
                _, loss = model(x, y)
            else:
                # Targets are already shifted. Passing them through HF's labels
                # interface would shift again and score the wrong token pairs.
                logits = model(x, use_cache=False).logits
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite held-out loss.")
        count = y.numel()
        nll_sum += loss.item() * count
        scored_tokens += count
    loss = nll_sum / scored_tokens
    result = {
        "checkpoint": args.checkpoint.name if args.checkpoint else "openai-community/gpt2",
        "reference_revision": reference_revision,
        "step": checkpoint["step"] if checkpoint is not None else None,
        "training_tokens": checkpoint["tokens_seen"] if checkpoint is not None else None,
        "loss": loss,
        "perplexity": math.exp(loss),
        "scored_tokens": scored_tokens,
        "evaluated_windows": limit,
        "available_windows": window_count,
        "complete_window_sweep": limit == window_count,
        "unscored_token_tail": len(tokens) - 1 - window_count * config.block_size,
        "context_length": config.block_size,
        "evaluation_protocol": "non-overlapping contexts; one-token-ahead targets; document boundaries unmasked",
        "tokenizer": meta["tokenizer"],
        "validation_sha256": meta["sha256"]["val"],
        "precision": "fp32" if device.type != "cuda" else str(dtype),
        "seconds": time.perf_counter() - begin,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
