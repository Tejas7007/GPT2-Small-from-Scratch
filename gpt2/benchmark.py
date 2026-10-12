"""Measure training speed and compare microbatch configurations.

Each candidate runs in a separate directory with the original token target
and learning-rate schedule. Compiler warmup is excluded from steady-state
update timing and included in the observed wall time. The timing projection
includes measured validation and checkpoint overhead.
"""

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def project_run(records, manifest, warmup_steps):
    """Project total time from complete update samples after warmup.

    Mean update time converts the planned update count into a total duration.
    Include periodic validation and milestone checkpoint writes once each.
    """
    updates = [
        row["seconds"]
        for row in records
        if row.get("kind") == "train" and row["step"] > warmup_steps
    ]
    if not updates or any(not math.isfinite(value) or value <= 0 for value in updates):
        raise ValueError("Benchmark needs finite positive update times after warmup.")
    validation = [row["validation_seconds"] for row in records if row.get("kind") == "val"]
    checkpoints = [row["checkpoint_seconds"] for row in records if row.get("kind") == "checkpoint"]
    if not validation or not checkpoints:
        raise ValueError("Benchmark is missing validation or checkpoint measurements.")
    if any(not math.isfinite(value) or value < 0 for value in validation + checkpoints):
        raise ValueError("Validation/checkpoint measurements must be finite and nonnegative.")
    steps = manifest["total_steps"]
    interval = manifest["args"]["eval_interval"]
    tokens_per_step = manifest["tokens_per_step"]
    evaluation_steps = set(range(interval, steps + 1, interval)) | {steps}
    evaluation_steps.update(
        math.ceil(value / tokens_per_step)
        for value in manifest["args"]["milestone_tokens"]
        if value <= steps * tokens_per_step
    )
    mean_seconds = statistics.mean(updates)
    training_seconds = steps * mean_seconds
    validation_seconds = (1 + len(evaluation_steps)) * max(validation)
    checkpoint_seconds = len(evaluation_steps) * max(checkpoints)
    checkpoint_interval = manifest["args"].get("checkpoint_interval", 0)
    if checkpoint_interval:
        checkpoint_steps = evaluation_steps | set(
            range(checkpoint_interval, steps + 1, checkpoint_interval)
        )
        checkpoint_seconds = len(checkpoint_steps) * max(checkpoints)
    total_seconds = training_seconds + validation_seconds + checkpoint_seconds
    return {
        "measured_updates": len(updates),
        "mean_update_seconds": mean_seconds,
        "median_update_seconds": statistics.median(updates),
        "global_tokens_per_second": tokens_per_step / mean_seconds,
        "projected_training_hours": training_seconds / 3600,
        "projected_validation_hours": validation_seconds / 3600,
        "projected_checkpoint_hours": checkpoint_seconds / 3600,
        "projected_gpu_session_hours": total_seconds / 3600,
    }


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/h100-4gpu.json")
    parser.add_argument("--gpus", type=int, choices=[1, 2, 4, 8], default=4)
    parser.add_argument("--microbatches", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--global-batch-tokens", type=int, default=524288)
    parser.add_argument("--validation-tokens", type=int, default=10485760)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--measure-steps", type=int, default=20)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    if min(args.microbatches) < 1 or args.warmup_steps < 1 or args.measure_steps < 2:
        parser.error("Use positive microbatches/warmup and at least two measured updates.")
    if args.global_batch_tokens < 1 or args.validation_tokens < 1:
        parser.error("Token counts must be positive.")
    if args.device == "cpu" and args.gpus != 1:
        parser.error("CPU verification uses --gpus 1.")
    if os.environ.get("WORLD_SIZE", "1") != "1":
        parser.error("Run python -m gpt2.benchmark directly; it launches torchrun itself.")
    return args


def main():
    args = arguments()
    defaults = json.loads(args.config.read_text())
    block_size = 64 if defaults.get("model", "gpt2") == "tiny" else 1024
    for batch in args.microbatches:
        batch_tokens = batch * block_size * args.gpus
        if args.global_batch_tokens % batch_tokens:
            raise SystemExit(f"Microbatch {batch} does not divide the global token batch.")
    total_steps = math.ceil(defaults["tokens"] / args.global_batch_tokens)
    stop_after = args.warmup_steps + args.measure_steps
    if total_steps < stop_after:
        raise SystemExit("The configured token target is too short for this benchmark.")
    output = args.out or ROOT / "runs/benchmarks" / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    begin = time.perf_counter()
    candidates = []
    for batch in dict.fromkeys(args.microbatches):
        directory = output / f"batch-{batch}"
        directory.mkdir()
        validation_batches = max(
            1, math.ceil(args.validation_tokens / (batch * block_size * args.gpus))
        )
        command = [
            sys.executable,
            "-m",
            "gpt2.train",
            "--config",
            str(args.config.resolve()),
            "--data",
            str(args.data.resolve()),
            "--out",
            str(directory),
            "--batch-size",
            str(batch),
            "--grad-accum",
            str(args.global_batch_tokens // (batch * block_size * args.gpus)),
            "--eval-batches",
            str(validation_batches),
            "--log-interval",
            "1",
            "--stop-after",
            str(stop_after),
            "--device",
            args.device,
            "--compile" if args.compile else "--no-compile",
        ]
        if args.device == "cpu":
            command += ["--precision", "fp32"]
        if args.gpus > 1:
            # torchrun's --module option imports the same package entry point
            # used by single-device execution. Each candidate retains its full
            # configured learning-rate schedule despite stopping after timing.
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={args.gpus}",
                "--module",
                *command[2:],
            ]
        print(f"Benchmarking batch={batch}; details: {directory / 'console.log'}", flush=True)
        candidate_begin = time.perf_counter()
        with (directory / "console.log").open("w") as console:
            result = subprocess.run(command, cwd=ROOT, stdout=console, stderr=subprocess.STDOUT)
        candidate = {
            "batch_size": batch,
            "directory": str(directory),
            "returncode": result.returncode,
            "wall_seconds": time.perf_counter() - candidate_begin,
        }
        if result.returncode == 0:
            manifest = json.loads((directory / "manifest.json").read_text())
            records = [
                json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()
            ]
            candidate.update(
                project_run(
                    records,
                    manifest,
                    args.warmup_steps,
                )
            )
            candidate["peak_cuda_memory_bytes"] = (
                max((row.get("peak_cuda_memory_bytes") or 0 for row in records), default=0) or None
            )
        else:
            candidate["error_log"] = str(directory / "console.log")
        candidates.append(candidate)
        print(json.dumps(candidate), flush=True)
    wall_seconds = time.perf_counter() - begin
    successful = [candidate for candidate in candidates if candidate["returncode"] == 0]
    for candidate in successful:
        candidate["projected_total_runtime_hours"] = (
            candidate["projected_gpu_session_hours"] + wall_seconds / 3600
        )
    best = (
        min(successful, key=lambda row: row["projected_gpu_session_hours"]) if successful else None
    )
    report = {
        "candidates": candidates,
        "recommended_batch_size": None if best is None else best["batch_size"],
        "benchmark_wall_seconds": wall_seconds,
        "gpu_performance_measured": args.device == "cuda" and best is not None,
        "data_preparation_included": False,
    }
    (output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    if best is not None:
        (output / "launch.env").write_text(
            f"export TRAIN_MICROBATCH={best['batch_size']}\nexport TRAIN_COMPILE={int(args.compile)}\n"
        )
    print(f"Report: {output / 'benchmark.json'}", flush=True)
    if best is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
