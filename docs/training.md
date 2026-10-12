# Training protocol

The run trained a randomly initialized GPT-2 Small model on FineWeb, using
four H100 GPUs and a target of ten billion processed tokens. It completed
19,074 optimizer updates and processed 10,000,269,312 tokens. The small
overshoot comes from finishing the final 524,288-token batch.

The training process starts with two token files, one for training and one
for validation. The loader draws context windows from them, the model
predicts each window's next tokens, and AdamW updates its parameters from
the resulting loss. Periodic validation measures progress on the same
sampled windows, while checkpoints let training resume after an interruption.

## Preparing the corpus

The corpus comes from
[FineWeb sample-10BT](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
at revision `9bb295ddab0e05d785b879661af7260fed5140fc`. Preparation reads
fourteen Parquet shards in source order and processes 14,357,844 documents.
It produces 9,900,003,494 training tokens and 99,997,824 validation tokens,
including document separators.

Each document is assigned to a split by hashing its UTF-8 text with BLAKE2b.
Hashes falling in the first 1% of the hash range go to validation. Identical
text receives the same assignment, so exact duplicate documents stay in
one split. Document lengths determine the eventual token ratio. FineWeb
supplies the corpus filtering and deduplication; additional near-duplicate
filtering and benchmark decontamination were outside this run.

The GPT-2 tokenizer converts each document to IDs and appends an end-of-text
ID. The preparation script writes them as little-endian unsigned 16-bit
values. This format fits the 50,257-token vocabulary in two bytes per token
and lets the loader read small portions of the corpus through a memory map.

```bash
python -m pip install -r requirements-data.txt
python -m gpt2.prepare_data \
  --hf-dataset HuggingFaceFW/fineweb \
  --hf-config sample-10BT \
  --hf-revision 9bb295ddab0e05d785b879661af7260fed5140fc \
  --out data/fineweb10b \
  --tokenizer gpt2 \
  --max-tokens 10000000000
```

The token cap includes both splits. Preparation finishes the document that
crosses the cap, which accounts for the few extra tokens in the corpus.
Parquet is read in bounded batches, and parallel tokenization returns the
documents in source order.

Preparation can resume at a completed document boundary after an interruption.
The script skips documents it has already written and continues tokenizing
from that point.

## Building a distributed batch

Training reads non-overlapping windows of 1,024 input tokens. Each window
includes one additional source token so the targets can be shifted one
position ahead. Packed windows can contain several documents, with
end-of-text tokens marking their boundaries. The causal mask spans the
whole window.

Each of the four training processes owns one GPU. These processes are called
ranks. At the start of an epoch, every rank reconstructs the same seeded
permutation of window indices. They take different slices of that permutation,
giving each GPU its own sixty-four sequences for a microbatch. Leftover
windows that cannot form a complete global microbatch are skipped at the
epoch boundary, along with an incomplete context tail.

Each GPU processes two microbatches before an optimizer update. Across the
four GPUs, that gives

```math
64 \times 1024 \times 2 \times 4 = 524{,}288
```

tokens per update. The ten-billion-token target exceeds the prepared training
stream, so the loader enters another epoch and revisits some windows near
the end. The processed-token count includes those revisits.

## Taking an optimizer step

The trainer divides each microbatch loss by two before backpropagation.
Gradients from the first microbatch accumulate locally; DistributedDataParallel
synchronizes them during the second and averages them across the four ranks.
The resulting gradient is the mean over the full 524,288-token batch.

AdamW tracks moving averages of gradients and their squared values to set
the update for each parameter. Weight decay is applied to matrices, including
embeddings, while biases and LayerNorm parameters use zero decay. Before the
step, the trainer measures the global gradient norm and clips it to a maximum
of 1.0. The logged norm is the value before clipping, so the plot can show
values above 1.0. A nonfinite norm stops the run.

The learning rate rises to 6e-4 over 700 warmup updates, then follows a cosine
curve down to 6e-5. CUDA matrix operations use bfloat16 autocast while model
parameters remain in full precision. SDPA, fused AdamW, and `torch.compile`
provide the accelerated execution path used on the H100 node.

| Setting | Run configuration |
| --- | --- |
| Model | 124,439,808 unique parameters |
| Layers, heads, width | 12, 12, 768 |
| Context and vocabulary | 1,024 tokens and 50,257 GPT-2 BPE IDs |
| Hardware | Four NVIDIA H100 80GB HBM3 GPUs |
| Software | Python 3.12.3 and PyTorch 2.8.0+cu128 |
| Microbatch and accumulation | 64 sequences per GPU, two microbatches per update |
| Effective token batch | 524,288 |
| Requested training tokens | 10,000,000,000 |
| Optimizer updates | 19,074 |
| Optimizer | Fused CUDA AdamW |
| AdamW betas and epsilon | 0.9, 0.95 and 1e-8 |
| Learning-rate schedule | 700-update warmup to 6e-4, cosine decay to 6e-5 |
| Matrix weight decay | 0.1 |
| Gradient clipping | Global norm maximum 1.0 |
| Precision | bfloat16 autocast, full-precision model parameters |
| Attention and compilation | PyTorch SDPA and `torch.compile` |
| Dropout and seed | Disabled and 1337 |
| Periodic validation | 10,485,760 sampled targets every 1,000 updates |
| Snapshot milestones | 100M, 1B, 2.5B, 5B and 10B processed tokens |

## Validation and checkpoint selection

At each scheduled validation, four GPUs each process forty microbatches of
sixty-four sequences. This scores 10,485,760 next-token targets. Resetting the
validation seed reuses the same sampled windows at each measurement, making
the curve easier to interpret as training progresses. Validation also runs
when a snapshot milestone is crossed and at the final update.

The trainer writes `best.pt` whenever this sampled loss improves. For this
run, the lowest value was 3.2684104 at step 19,000, after 9,961,472,000
processed tokens. Training then finished at step 19,074 with sampled loss
3.2685382. The final held-out evaluation, HellaSwag evaluation, and prompt
suite all use the selected step-19,000 checkpoint. The
[evaluation guide](evaluation.md) explains the full held-out sweep and the
reference comparison.

## Starting and continuing a run

Run these commands from the repository root using
[the four-GPU configuration](../configs/h100-4gpu.json).

```bash
python -m pip install -r requirements.txt
torchrun --standalone --nproc_per_node=4 --module gpt2.train \
  --config configs/h100-4gpu.json
```

A fresh run needs a fresh output directory. To continue an existing run,
pass its latest full checkpoint.

```bash
torchrun --standalone --nproc_per_node=4 --module gpt2.train \
  --config configs/h100-4gpu.json \
  --resume runs/gpt2-fineweb10b-h100-4gpu/last.pt
```

The checkpoint restores the model, optimizer, precision scaler, each rank's
random state, and its position in the training stream. The learning-rate
schedule continues from the saved update. Resume requires the original
architecture, GPU count, microbatch and accumulation sizes, token target,
optimizer settings, seed, precision, and dataset. The loader then continues
with the next batch from its saved position.

## Checkpoints and learning curves

| File | Contents | Use |
| --- | --- | --- |
| `last.pt` | Model, optimizer, scaler, random and sampler states | Continue from the latest checkpoint |
| `best.pt` | Full state at the lowest sampled validation loss | Evaluate the selected model |
| `snapshots/step_*.pt` | Model weights, dimensions, data metadata and token count | Compare checkpoints across training milestones |
| `metrics.jsonl` | Training, validation, checkpoint and completion events | Plot learning and runtime |

Checkpoints are written every 1,000 updates and at validation milestones.
The save routine serializes on fast local storage, copies to a temporary file
at the destination, flushes it, and replaces the previous checkpoint name.
A distributed barrier makes every rank wait for that write before timing the
next update.

SIGINT and SIGTERM request a stop after a complete optimizer update, followed
by validation and a checkpoint. Recovery after an abrupt power loss starts
from the most recent completed checkpoint. Milestone snapshots contain model
weights and metadata, so continuing optimization uses `last.pt` or `best.pt`,
which also contain the optimizer and sampler states.

CUDA kernels, hardware, and software versions can introduce floating-point
differences in a rerun, even when the seed and data are the same.

[Runtime measurement](benchmark.md) explains the timing fields, and
[the walkthrough](walkthrough.md) follows the implementation in reading order.
