"""Checkpointed causal pretraining on one device or a torchrun process group.

Effective token batches are formed with gradient accumulation. Configuration,
data identity, optimizer/scaler state, and per-rank sampling state are retained
to support reproducible continuation and measured training reports.
"""

import argparse
import json
import math
import os
import signal
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from gpt2.checkpoint import atomic_save, file_sha256
from gpt2.data import TokenStream
from gpt2.model import GPT, GPTConfig
from gpt2.provenance import provenance


def arguments():
    """Overlay explicit flags on a validated JSON configuration.

    Parse the config path first so argparse can supply its values as defaults.
    A second parse handles the complete CLI, giving explicit flags precedence.
    Resume and interruption controls belong to a session rather than a recipe.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, help="JSON defaults; explicit CLI flags take precedence")
    p.add_argument("--data", type=Path)
    p.add_argument("--out", type=Path, default=Path("runs/baseline"))
    p.add_argument("--model", choices=["gpt2", "tiny"], default="gpt2")
    p.add_argument("--batch-size", type=int, default=4, help="Sequences per GPU/microbatch")
    p.add_argument("--grad-accum", type=int, default=8, help="Microbatches per optimizer update")
    p.add_argument(
        "--tokens", type=int, default=1_000_000, help="Approximate total training-token budget"
    )
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--warmup-steps", type=int, default=700)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--eval-batches", type=int, default=10)
    p.add_argument("--checkpoint-interval", type=int, default=100)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="auto", help="auto, cpu, mps or cuda")
    p.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"], default="auto")
    p.add_argument("--attention", choices=["manual", "sdpa"], default="sdpa")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--resume", type=Path)
    p.add_argument(
        "--stop-after",
        type=int,
        help="Stop at this absolute update, keeping the original lr schedule",
    )
    p.add_argument(
        "--verify-data", action="store_true", help="Recompute token-file SHA-256 before training"
    )
    p.add_argument(
        "--milestone-tokens",
        nargs="*",
        type=int,
        default=[],
        help="Token counts for retained model snapshots",
    )
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", type=Path)
    config_path = preliminary.parse_known_args()[0].config
    if config_path:
        defaults = json.loads(config_path.read_text())
        valid = {action.dest for action in p._actions} - {"help", "config", "resume", "stop_after"}
        if not isinstance(defaults, dict) or set(defaults) - valid:
            p.error("Config must be a JSON object containing supported training argument names.")
        p.set_defaults(**defaults)
    args = p.parse_args()
    if args.data is None:
        p.error("Provide --data or a data directory in the configuration.")
    positive = (
        args.batch_size,
        args.grad_accum,
        args.tokens,
        args.eval_interval,
        args.eval_batches,
        args.log_interval,
        args.checkpoint_interval,
    )
    if min(positive) < 1 or args.lr <= 0 or args.weight_decay < 0 or args.warmup_steps < 0:
        p.error("Counts/lr must be positive; warmup and weight decay must be nonnegative.")
    if args.stop_after is not None and args.stop_after < 1:
        p.error("stop-after must be positive.")
    if any(value < 1 for value in args.milestone_tokens):
        p.error("milestone-tokens must be positive.")
    args.milestone_tokens = sorted(set(args.milestone_tokens))
    return args


def setup_device(args, world, local_rank):
    """Bind each torchrun worker to one device and initialize its process group.

    RANK identifies the worker globally, while LOCAL_RANK indexes the visible
    devices on one host. CUDA workers use NCCL; CPU verification uses Gloo.
    """
    name = args.device
    if name == "auto":
        name = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    if name.startswith("cuda"):
        device = (
            torch.device("cuda", local_rank) if world > 1 or name == "cuda" else torch.device(name)
        )
        torch.cuda.set_device(device)
    else:
        device = torch.device(name)
    if world > 1:
        if device.type == "mps":
            raise ValueError("Distributed MPS training is not supported.")
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    return device


def learning_rate(step, total, warmup, maximum):
    """Return the zero-based warmup/cosine schedule for the full token budget.

    An interrupted session retains ``total`` so its early updates are identical
    to uninterrupted training. The last scheduled update reaches 10% of peak.
    """
    # step is zero-based. Warm up gently, then decay toward 10% of max lr.
    if step < warmup:
        return maximum * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup - 1)
    return maximum * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1))))


def mean_across_ranks(value, world):
    """Average an equal-sized per-rank scalar in place.

    All workers process the same number of tokens per microbatch, so averaging
    their mean losses equals the mean over the complete distributed batch.
    """
    if world > 1:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= world
    return value


def main():
    """Validate corpus identity, train or resume, and retain measured artifacts."""
    args = arguments()
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    device = setup_device(args, world, local_rank)
    session_begin = time.perf_counter()
    torch.manual_seed(args.seed)  # identical initial weights on every rank
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    meta = json.loads((args.data / "meta.json").read_text())
    config = (
        GPTConfig(attention=args.attention)
        if args.model == "gpt2"
        else GPTConfig(
            vocab_size=meta["vocab_size"],
            block_size=64,
            n_layer=2,
            n_head=4,
            n_embd=128,
            attention=args.attention,
        )
    )
    if config.vocab_size != meta["vocab_size"]:
        raise ValueError("GPT-2 requires GPT-2 tokenization. Use --model tiny for byte demo data.")
    for split in ("train", "val"):
        expected = meta["tokens"][split] * 2
        if (args.data / f"{split}.bin").stat().st_size != expected:
            raise ValueError(f"{split}.bin size differs from metadata.")
        if args.verify_data and rank == 0:
            if file_sha256(args.data / f"{split}.bin") != meta["sha256"][split]:
                raise ValueError(f"{split}.bin digest differs from metadata.")
    if world > 1:
        dist.barrier()

    precision = args.precision
    if precision == "auto":
        precision = (
            ("bf16" if torch.cuda.is_bf16_supported() else "fp16")
            if device.type == "cuda"
            else "fp32"
        )
    if precision != "fp32" and device.type != "cuda":
        raise ValueError("This trainer uses mixed precision only on CUDA; use fp32 here.")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device does not support bf16; use auto or fp16.")
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    def autocast():
        return torch.autocast("cuda", dtype=dtype) if precision != "fp32" else nullcontext()

    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")

    raw = GPT(config).to(device)
    # Weight matrices decay; biases and LayerNorm's one-dimensional parameters
    # do not. parameters() counts the tied embedding/output parameter once.
    decay = [p for p in raw.parameters() if p.ndim >= 2]
    no_decay = [p for p in raw.parameters() if p.ndim < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=device.type == "cuda",
    )

    train_stream = TokenStream(
        args.data / "train.bin",
        config.block_size,
        args.batch_size,
        args.seed,
        device,
        rank=rank,
        world_size=world,
    )
    val_stream = TokenStream(
        args.data / "val.bin",
        config.block_size,
        args.batch_size,
        args.seed + 100_000,
        device,
        mode="random",
        rank=rank,
        world_size=world,
    )
    # Effective batch = sequences * tokens/sequence * microbatches * GPUs.
    tokens_per_step = args.batch_size * config.block_size * args.grad_accum * world
    total_steps = math.ceil(args.tokens / tokens_per_step)
    warmup = min(args.warmup_steps, max(0, total_steps // 10))
    # Cap warmup at 10% of updates for short runs; preserve the full configured
    # warmup for the baseline token budget. Record the effective value explicitly.
    start_step, best_val = 0, float("inf")
    elapsed_before_session = 0.0
    if args.resume:
        # Continuation is stricter than loading weights. Changing batch layout,
        # optimizer schedule, precision, corpus, or rank count would make saved
        # optimizer moments and sample cursors describe a different trajectory.
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 2 or "optimizer" not in checkpoint:
            raise ValueError(
                "Resume requires a version-2 training checkpoint, not a model snapshot."
            )
        if checkpoint["config"] != asdict(config) or checkpoint["world_size"] != world:
            raise ValueError("Resume requires the same model config and number of ranks.")
        for key in (
            "batch_size",
            "grad_accum",
            "tokens",
            "lr",
            "warmup_steps",
            "weight_decay",
            "seed",
            "precision",
        ):
            if checkpoint["args"][key] != vars(args)[key]:
                raise ValueError(f"Resume requires the original --{key.replace('_', '-')} value.")
        if checkpoint["data_meta"] != meta or checkpoint["precision"] != precision:
            raise ValueError("Resume data metadata/precision differs from the original run.")
        raw.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        state = checkpoint["rank_states"][rank]
        train_stream.load_state_dict(state["train_state"])
        val_stream.load_state_dict(state["val_state"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"], device)
        start_step, best_val = checkpoint["step"], checkpoint["best_val"]
        elapsed_before_session = checkpoint.get("elapsed_seconds", 0.0)

    model = torch.compile(raw) if args.compile else raw
    if world > 1:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
    args.out.mkdir(parents=True, exist_ok=True)
    provenance_record = provenance()
    if rank == 0:
        if not args.resume and (args.out / "metrics.jsonl").exists():
            raise ValueError("Run directory already has metrics; choose a new --out or resume.")
        manifest = {
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "config": asdict(config),
            "parameters": raw.parameter_count(),
            "data_meta": meta,
            "torch_version": str(torch.__version__),
            "device": str(device),
            "hardware": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else device.type,
            "precision": precision,
            "world_size": world,
            "tokens_per_step": tokens_per_step,
            "total_steps": total_steps,
            "actual_warmup_steps": warmup,
            "provenance": provenance_record,
            "sampling": "rank-disjoint shuffled fixed-length windows",
        }
        if not args.resume:
            (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        else:
            # Retain only metrics covered by the resumed checkpoint. This makes
            # continuation from an older checkpoint a coherent single trajectory.
            metric_path = args.out / "metrics.jsonl"
            if metric_path.exists():
                records = [
                    json.loads(line) for line in metric_path.read_text().splitlines() if line
                ]
                records = [row for row in records if row.get("step", 0) <= start_step]
                metric_path.write_text("".join(json.dumps(row) + "\n" for row in records))
        with open(args.out / "sessions.jsonl", "a") as handle:
            handle.write(
                json.dumps(
                    {
                        "started_unix": time.time(),
                        "event": "start",
                        "resume_step": start_step,
                        "provenance": provenance_record,
                        "precision": precision,
                    }
                )
                + "\n"
            )
        print(
            f"parameters={raw.parameter_count():,}; tokens/update={tokens_per_step:,}; updates={total_steps:,}",
            flush=True,
        )

    @torch.no_grad()
    def evaluate():
        """Estimate validation loss on an identical sampled panel at every step."""
        raw.eval()
        # Reuse the same validation windows at every evaluation. The standalone
        # evaluator performs a contiguous held-out sweep after training.
        val_stream.generator.manual_seed(args.seed + 100_000 + rank)
        loss_sum = torch.zeros((), device=device)
        for _ in range(args.eval_batches):
            x, y = val_stream.batch()
            with autocast():
                _, loss = raw(x, y)
            loss_sum += loss.float()
        raw.train()
        return mean_across_ranks(loss_sum / args.eval_batches, world).item()

    def save(step, val_loss, improved, retain_snapshot=False):
        """Save complete continuation state and optional compact milestone weights.

        Every rank contributes its sampler cursors and RNG state, even though
        only rank zero writes shared model/optimizer tensors. Restore these
        local states before the next batch to prevent skipped or repeated data.
        """
        state = {
            "train_state": train_stream.state_dict(),
            "val_state": val_stream.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        }
        states = [None] * world
        if world > 1:
            dist.all_gather_object(states, state)
        else:
            states = [state]
        if rank == 0:
            payload = {
                "format_version": 2,
                "config": asdict(config),
                "model": raw.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step,
                "tokens_seen": step * tokens_per_step,
                "elapsed_seconds": elapsed_before_session + time.perf_counter() - session_begin,
                "best_val": best_val,
                "val_loss": val_loss,
                "world_size": world,
                "rank_states": states,
                "data_meta": meta,
                "precision": precision,
                "provenance": provenance_record,
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            }
            atomic_save(payload, args.out / "last.pt")
            if improved:
                atomic_save(payload, args.out / "best.pt")
            if retain_snapshot:
                directory = args.out / "snapshots"
                directory.mkdir(exist_ok=True)
                # Milestones retain weights and provenance without duplicating
                # optimizer tensors. last.pt/best.pt remain resumable checkpoints.
                snapshot = {
                    key: payload[key]
                    for key in (
                        "format_version",
                        "config",
                        "model",
                        "step",
                        "tokens_seen",
                        "val_loss",
                        "data_meta",
                        "precision",
                        "provenance",
                    )
                }
                atomic_save(snapshot, directory / f"step_{step:08d}.pt")
        if world > 1:
            # Complete the shared checkpoint before any rank starts timing the
            # next update. Faster ranks otherwise charge file I/O to training.
            dist.barrier()

    def log(record):
        if rank == 0:
            with open(args.out / "metrics.jsonl", "a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)

    def validate_and_checkpoint(step, retain_snapshot=False):
        """Measure validation separately from checkpoint I/O and update best.pt.

        Model selection uses the fixed sampled validation panel. The final
        contiguous evaluation is a separate report and does not select weights.
        """
        nonlocal best_val, last_val_loss
        validation_begin = time.perf_counter()
        val = evaluate()
        last_val_loss = val
        validation_seconds = time.perf_counter() - validation_begin
        if not math.isfinite(val):
            raise FloatingPointError("Nonfinite validation loss.")
        improved = val < best_val
        best_val = min(val, best_val)
        log(
            {
                "kind": "val",
                "step": step,
                "tokens_seen": step * tokens_per_step,
                "loss": val,
                "perplexity": math.exp(min(val, 80)),
                "best_loss": best_val,
                "validation_seconds": validation_seconds,
            }
        )
        checkpoint_begin = time.perf_counter()
        save(step, val, improved, retain_snapshot=retain_snapshot)
        log(
            {
                "kind": "checkpoint",
                "step": step,
                "checkpoint_seconds": time.perf_counter() - checkpoint_begin,
            }
        )

    stop_requested = {"value": False}
    stop_file = args.out / "stop.request"

    def stopping():
        # Reduce the stop decision across workers. A signal delivered to one
        # rank must not let other ranks enter another unmatched DDP collective.
        stop = torch.tensor(int(stop_requested["value"] or stop_file.is_file()), device=device)
        if world > 1:
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        return bool(stop.item())

    def request_stop(signum, frame):
        # Defer checkpointing to the update boundary; signal handlers must not
        # interrupt tensor collectives or serialize a partially updated optimizer.
        stop_requested["value"] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    raw.train()
    last_val_loss = checkpoint["val_loss"] if args.resume else None
    if start_step == 0:
        validation_begin = time.perf_counter()
        initial = evaluate()
        last_val_loss = initial
        if not math.isfinite(initial):
            raise FloatingPointError("Nonfinite initial validation loss.")
        log(
            {
                "kind": "val",
                "step": 0,
                "tokens_seen": 0,
                "loss": initial,
                "perplexity": math.exp(min(initial, 80)),
                "validation_seconds": time.perf_counter() - validation_begin,
            }
        )
    run_end = min(args.stop_after, total_steps) if args.stop_after is not None else total_steps
    finished_step = start_step
    if stopping():
        validate_and_checkpoint(start_step)
        run_end = start_step
    for step in range(start_step, run_end):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        begin = time.perf_counter()
        lr = learning_rate(step, total_steps, warmup, args.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = torch.zeros((), device=device)
        for micro in range(args.grad_accum):
            x, y = train_stream.batch()
            # Synchronize gradients only on the last microbatch in DDP. The
            # context must include both forward and backward.
            sync = model.no_sync() if world > 1 and micro < args.grad_accum - 1 else nullcontext()
            with sync:
                with autocast():
                    _, loss = model(x, y)
                accumulated_loss += loss.detach().float() / args.grad_accum
                # Normalize accumulated gradients to the effective batch mean.
                # DDP subsequently averages these gradients across all ranks.
                scaler.scale(loss / args.grad_accum).backward()
        scaler.unscale_(optimizer)
        # Clip the true accumulated gradient after undoing FP16's scale. BF16
        # uses the same path with a disabled scaler. Report the norm before
        # clipping so stability diagnostics do not hide threshold crossings.
        norm = torch.nn.utils.clip_grad_norm_(raw.parameters(), max_norm=1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite gradient norm; stop and inspect precision/lr/data.")
        scaler.step(optimizer)
        scaler.update()
        loss_value = mean_across_ranks(accumulated_loss, world).item()
        elapsed = time.perf_counter() - begin
        should_stop = stopping()
        crossed_milestone = any(
            step * tokens_per_step < value <= (step + 1) * tokens_per_step
            for value in args.milestone_tokens
        )
        if (
            step == start_step
            or (step + 1) % args.log_interval == 0
            or step + 1 == run_end
            or should_stop
        ):
            log(
                {
                    "kind": "train",
                    "step": step + 1,
                    "tokens_seen": (step + 1) * tokens_per_step,
                    "loss": loss_value,
                    "lr": lr,
                    "grad_norm": norm.item(),
                    "seconds": elapsed,
                    "elapsed_seconds": elapsed_before_session + time.perf_counter() - session_begin,
                    "tokens_per_second": tokens_per_step / elapsed,
                    "epoch": train_stream.epoch,
                    "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda"
                    else None,
                }
            )
        if (
            (step + 1) % args.eval_interval == 0
            or step + 1 == run_end
            or crossed_milestone
            or should_stop
        ):
            validate_and_checkpoint(step + 1, retain_snapshot=crossed_milestone)
        elif (step + 1) % args.checkpoint_interval == 0:
            checkpoint_begin = time.perf_counter()
            save(step + 1, last_val_loss, False)
            log(
                {
                    "kind": "checkpoint",
                    "step": step + 1,
                    "checkpoint_seconds": time.perf_counter() - checkpoint_begin,
                }
            )
        if should_stop:
            finished_step = step + 1
            break
        finished_step = step + 1
    if rank == 0:
        with open(args.out / "sessions.jsonl", "a") as handle:
            handle.write(
                json.dumps(
                    {
                        "event": "end",
                        "finished_unix": time.time(),
                        "step": finished_step,
                        "tokens_seen": finished_step * tokens_per_step,
                        "complete": finished_step >= total_steps,
                        "wall_seconds": time.perf_counter() - session_begin,
                    }
                )
                + "\n"
            )
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
