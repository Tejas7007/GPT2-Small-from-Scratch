"""Zero-shot HellaSwag validation using ending-only next-token likelihoods.

Context preprocessing follows EleutherAI's HellaSwag task: activity label,
ctx_a, capitalized ctx_b, and removal of WikiHow bracket artifacts. Report
raw, token-normalized, and character-normalized accuracy separately. Exact
comparisons should use the released GPT-2 reference evaluated by this script.
"""

import argparse
import hashlib
import json
import os
import re
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn import functional as F

from model import GPT, GPTConfig
from tokenizer import Tokenizer
from train import file_sha256

DATASET = "Rowan/hellaswag"
DATASET_REVISION = "218ec52e09a7e7462a5400043bb9a69a41d06b76"
VALIDATION_ROWS = 10042
EVALUATION_PROTOCOL = (
    "ending-only next-token likelihood; separate prefix and space-prefixed ending BPE; "
    "right padding masked; no truncation; first-index argmin ties"
)


def preprocess(text):
    """Remove the dataset's display artifacts without consulting the label."""
    text = text.strip().replace(" [title]", ". ")
    return re.sub(r"\[.*?\]", "", text).replace("  ", " ")


def encode_choices(row, tokenizer, block_size):
    """Return four token sequences and the first ending target's token index."""
    context = preprocess(
        row["activity_label"] + ": " + row["ctx_a"] + " " + row["ctx_b"].capitalize()
    )
    prefix = tokenizer.encode(context)
    endings = [preprocess(ending) for ending in row["endings"]]
    if not prefix or len(endings) != 4:
        raise ValueError("HellaSwag rows must have a nonempty context and four endings.")
    sequences = [prefix + tokenizer.encode(" " + ending) for ending in endings]
    if any(len(sequence) <= len(prefix) for sequence in sequences):
        raise ValueError("Every candidate must contain at least one ending token.")
    if max(map(len, sequences)) - 1 > block_size:
        raise ValueError("A HellaSwag candidate exceeds context length; no rows are truncated.")
    return sequences, len(prefix), [len(ending) for ending in endings]


def make_batch(choices, pad_id, device):
    """Right-pad causal inputs; mask all prompt and padding prediction targets."""
    width = max(len(sequence) - 1 for sequence, _ in choices)
    x = torch.full((len(choices), width), pad_id, dtype=torch.long, device=device)
    y = x.clone()
    mask = torch.zeros_like(x, dtype=torch.bool)
    for index, (sequence, first_ending) in enumerate(choices):
        length = len(sequence) - 1
        x[index, :length] = torch.tensor(sequence[:-1], device=device)
        y[index, :length] = torch.tensor(sequence[1:], device=device)
        # Logit position first_ending - 1 predicts the first ending token.
        mask[index, first_ending - 1 : length] = True
    if not torch.all(mask.any(dim=1)):
        raise ValueError("Every sequence needs a scored ending.")
    return x, y, mask


def ending_nll(logits, targets, mask):
    """Sum negative log likelihood over ending targets, excluding prompt/padding."""
    selected = F.cross_entropy(logits[mask].float(), targets[mask], reduction="none")
    losses = torch.zeros_like(targets, dtype=torch.float32)
    losses[mask] = selected
    sums = losses.sum(dim=1)
    counts = mask.sum(dim=1)
    if not torch.isfinite(sums).all():
        raise FloatingPointError("Nonfinite HellaSwag likelihood.")
    return sums, counts


def digest_rows(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--reference", choices=["gpt2"])
    parser.add_argument("--reference-revision", default="main")
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Rows per batch; four choices each"
    )
    parser.add_argument(
        "--max-examples", type=int, help="Optional explicitly partial first-N sweep"
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.max_examples is not None and args.max_examples < 1):
        parser.error("Batch size and optional example limit must be positive.")
    from datasets import load_dataset
    from huggingface_hub import HfApi

    resolved_revision = HfApi().dataset_info(DATASET, revision=args.dataset_revision).sha
    dataset = load_dataset(DATASET, revision=resolved_revision, split="validation")
    if resolved_revision == DATASET_REVISION and len(dataset) != VALIDATION_ROWS:
        raise ValueError("Pinned validation row count differs from the dataset card.")
    limit = min(len(dataset), args.max_examples or len(dataset))
    # Identity includes every validation row, even for an explicitly partial run.
    validation_digest = digest_rows(dataset)
    checkpoint = (
        torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if args.checkpoint
        else None
    )
    reference_revision = None
    if checkpoint is not None:
        if checkpoint["data_meta"]["tokenizer"] != "gpt2":
            raise ValueError("HellaSwag evaluation requires the GPT-2 tokenizer.")
        config = GPTConfig(**checkpoint["config"])
        if config.vocab_size != 50257:
            raise ValueError("HellaSwag evaluation requires the GPT-2 vocabulary.")
        model = GPT(config)
        model.load_state_dict(checkpoint["model"])
    else:
        from transformers import AutoModelForCausalLM

        identifier = "openai-community/gpt2"
        reference_revision = HfApi().model_info(identifier, revision=args.reference_revision).sha
        model = AutoModelForCausalLM.from_pretrained(identifier, revision=reference_revision)
        config = GPTConfig()
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    model.to(device).eval()
    tokenizer = Tokenizer("gpt2")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None
    # Greedy argmin needs no sampling RNG. Disable dropout and TF32; BF16 remains
    # the recorded CUDA inference precision and all likelihood reductions use FP32.
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.synchronize(device)
    begin = time.perf_counter()
    correct = {"sum": 0, "token_normalized": 0, "character_normalized": 0}
    scored_tokens = 0
    predictions_digest = hashlib.sha256()
    for first in range(0, limit, args.batch_size):
        rows = [dataset[index] for index in range(first, min(first + args.batch_size, limit))]
        choices, character_counts = [], []
        for row in rows:
            sequences, prefix_length, lengths = encode_choices(row, tokenizer, config.block_size)
            choices.extend((sequence, prefix_length) for sequence in sequences)
            character_counts.extend(lengths)
        x, y, mask = make_batch(choices, tokenizer.eot_id, device)
        autocast = torch.autocast("cuda", dtype=dtype) if dtype else nullcontext()
        with autocast:
            logits = model(x, y)[0] if checkpoint is not None else model(x, use_cache=False).logits
        sums, counts = ending_nll(logits, y, mask)
        sums, counts = sums.reshape(-1, 4), counts.reshape(-1, 4)
        characters = torch.tensor(character_counts, device=device).reshape(-1, 4)
        if not torch.all(characters > 0):
            raise ValueError("Empty HellaSwag ending after preprocessing.")
        predictions = {
            "sum": sums.argmin(dim=1).tolist(),
            "token_normalized": (sums / counts).argmin(dim=1).tolist(),
            "character_normalized": (sums / characters).argmin(dim=1).tolist(),
        }
        # Gold labels enter only after model scores and predictions are computed.
        for index, row in enumerate(rows):
            label = int(row["label"])
            if label not in range(4):
                raise ValueError("Validation gold label must be an index from zero to three.")
            for metric, predicted in predictions.items():
                correct[metric] += predicted[index] == label
            predictions_digest.update(
                json.dumps(
                    {
                        "index": first + index,
                        **{key: value[index] for key, value in predictions.items()},
                    },
                    sort_keys=True,
                ).encode("utf-8")
            )
            predictions_digest.update(b"\n")
        scored_tokens += counts.sum().item()
        if first == 0 or first + len(rows) == limit or (first // args.batch_size) % 100 == 0:
            print(f"HellaSwag {first + len(rows)}/{limit} rows", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - begin
    accuracies = {metric: value / limit for metric, value in correct.items()}
    result = {
        "task": "hellaswag",
        "split": "validation",
        "fewshot": 0,
        "dataset": DATASET,
        "dataset_revision": resolved_revision,
        "validation_rows_sha256": validation_digest,
        "dataset_fingerprint": dataset._fingerprint,
        "evaluated_examples": limit,
        "available_examples": len(dataset),
        "complete_validation_sweep": limit == len(dataset),
        "accuracy": accuracies["sum"],
        "accuracy_token_normalized": accuracies["token_normalized"],
        "accuracy_character_normalized": accuracies["character_normalized"],
        "correct_counts": correct,
        "standard_error": {
            key: (value * (1 - value) / limit) ** 0.5 for key, value in accuracies.items()
        },
        "chance_accuracy": 0.25,
        "scored_ending_tokens": scored_tokens,
        "checkpoint": args.checkpoint.name if args.checkpoint else "openai-community/gpt2",
        "checkpoint_sha256": file_sha256(args.checkpoint) if args.checkpoint else None,
        "reference_revision": reference_revision,
        "training_tokens": checkpoint["tokens_seen"] if checkpoint is not None else None,
        "step": checkpoint["step"] if checkpoint is not None else None,
        "tokenizer": "gpt2",
        "context_length": config.block_size,
        "precision": "bfloat16" if dtype else "float32",
        "device": str(device),
        "batch_size_examples": args.batch_size,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "preprocessing_source": "https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hellaswag/utils.py",
        "comparability": "Compare checkpoint and reference from this evaluator; tokenizer boundary conventions can differ from other harness versions.",
        "source_sha256": {
            name: file_sha256(Path(__file__).parent / name)
            for name in ["evaluate_hellaswag.py", "model.py", "tokenizer.py"]
        },
        "predictions_sha256": predictions_digest.hexdigest(),
        "seconds": seconds,
    }
    atomic_json(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
