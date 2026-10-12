# Measuring training speed

The benchmark helps choose a microbatch before the full run. Each candidate
uses the production trainer with the 124.4M-parameter model, 1,024-token
context, bfloat16 precision, AdamW settings, and the learning-rate schedule
planned for full training. Gradient accumulation adjusts to the microbatch so
each optimizer update still processes 524,288 tokens across four GPUs.

## Recorded measurements

[The four-GPU preflight report](../reports/preflight-h100-4gpu/staged-benchmark.json)
contains twenty measured updates after warmup. The selected microbatch of
sixty-four sequences per GPU uses two accumulation steps. Its mean update time
was 0.312 seconds, giving approximately 1.682M global tokens per second.

The checkpoint probe measured a 1,493,516,991-byte full checkpoint. Loading
it took 2.97 seconds and saving it to persistent storage took 4.57 seconds.
Model weights matched after the save and load. The
[CUDA continuation check](../reports/preflight-h100-4gpu/gpu_resume_verified.txt)
also compared distributed bfloat16 training with an interrupted and resumed
run.

These short measurements helped select the training profile. The complete run
also spends time compiling, evaluating validation batches, and writing
checkpoints.

## Benchmark another configuration

```bash
python -m gpt2.benchmark \
  --config configs/h100-4gpu.json \
  --data data/fineweb10b \
  --gpus 4 \
  --microbatches 32 64 \
  --out runs/benchmarks/h100-4gpu
```

Choose a fresh output directory for each benchmark. Each candidate writes its
own checkpoints, logs, and timing report. The default trial warms the compiler
and optimizer for ten updates, then measures twenty updates. These trials run
separately from the final model-quality experiment.

The report includes update timings, validation and checkpoint overhead,
benchmark wall time, and the recommended microbatch. The saved update records
also contain GPU memory measurements. The projected duration combines the
observed benchmark time with the estimated duration of all planned updates,
validation passes, and checkpoint writes. Corpus preparation is a separate
stage.

## Interpreting the completed run

The pilot and main trainer sessions total 6,456.047 seconds, or 107.6 minutes.
The pilot contributes 52.541 seconds and the main session contributes 6,403.506
seconds. These times include setup, training updates, and scheduled validation
and checkpoint writes within the two sessions. Dividing 10,000,269,312
processed tokens by that duration gives approximately 1.549M tokens per second
across the trainer sessions.

This duration covers the trainer sessions only. Corpus preparation, separate
benchmarks, final evaluation, and the gap between sessions fall outside it.
The steady-state update rate describes a narrower part of the workload.

| Measurement | Included work | Scope |
| --- | --- | --- |
| Logged update duration | Data transfer, forward and backward passes, accumulation, gradient synchronization, and optimizer update | Usually sampled every ten updates |
| Logged throughput | Global tokens per update divided by logged update duration | Update speed, with validation and checkpoint work timed separately |
| Validation duration | Scheduled sampled validation pass | A different workload from the final full evaluation |
| Checkpoint duration | Serialization, transfer to persistent storage, and distributed barrier | Depends on storage performance |
| Training-session wall time | Compiler startup, updates, scheduled validation, and checkpoint work | Time inside each trainer session |
| Peak CUDA allocation | PyTorch allocator's cumulative peak on rank zero | Allocated tensors on one GPU; reserved memory and other GPUs require separate measurements |

The plots show raw update durations without smoothing. Compilation and
continuation startup can make an early update much slower. Those observations
help explain the gap between steady-state speed and session time. Any
steady-state summary should state which startup observations it excludes.
Logged update timings provide a sample of training speed. Summing the pilot
and main training durations also accounts for startup, validation, and saves.

## Comparing runs

This experiment measures one four-H100 configuration. Measuring the individual
benefit of compilation, precision, attention backend, or optimizer settings
would require controlled comparisons. Measuring GPU scaling would require
equivalent runs on one, two, four, and eight devices. Those comparisons and
model FLOP utilization measurements are outside this experiment.

[Karpathy's 124M run](https://github.com/karpathy/llm.c/discussions/481) provides
context for the 10B-token target. Its implementation, hardware, and timing
scope differ. The training logs and runtime plots show the speed achieved here.
