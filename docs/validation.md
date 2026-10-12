# Testing the implementation

The test suite checks the model's math, the training batches, checkpoint
continuation, and evaluation scores. It uses small models and synthetic data
so most checks run on a CPU. A separate CUDA check exercises the distributed
training path on real GPUs.

## Model behavior

[The model tests](../tests/test_model.py) check that changing future input tokens
leaves earlier predictions unchanged. This is the defining constraint of
causal attention. They also compare the outputs and gradients of the manual
attention implementation with PyTorch's optimized scaled dot-product
attention.

Other tests check the 124,439,808 parameter count, shared input and output
vocabulary weights, and the one-token shift between inputs and targets. The
gradient accumulation test compares several microbatches with one larger
batch. Both should produce the same gradients because they represent the same
mean training objective. A small memorization test checks that optimization
can learn a repeated sequence.

## Data preparation and batching

[Sampler tests](../tests/test_stream.py) check that GPU workers receive distinct
windows within a global batch. They save the sampler's position, cross an
epoch boundary, and restore it to check that the next batch is reproducible.

Preparation tests compare interrupted and uninterrupted tokenization. They
also start two writers for the same output directory to check that one waits
for the other. A completed document must reach the token files before the
preparation progress file records it. That ordering lets preparation recover
from an interruption without skipping a partially written document.

## Checkpoint continuation

The end-to-end CPU test prepares a small corpus and trains it in two ways.
One run finishes uninterrupted. The other stops midway and resumes from its
checkpoint. The final model weights and optimizer state must agree. The test
then exercises milestone snapshots, held-out evaluation, text generation,
and plotting.

Checkpoint save tests interrupt a file copy and check that the previous
checkpoint can still be loaded. They also save and load shared tensors to check
that the checkpoint format retains the model's tied weights.

The CUDA continuation check repeats the interrupted-run comparison with
bfloat16 training across four GPUs. It compares model weights, optimizer
state, and each worker's sampling state. The recorded outcome is available in
[the preflight report](../reports/preflight-h100-4gpu).

## Evaluation scores

HellaSwag tests check where an ending begins and which prediction targets
contribute to its score. Prompt tokens and padding must have no effect on the
ending likelihood. Tests also show how dividing by ending length can change
the selected answer and check that adding the correct-answer label leaves the
encoded candidates unchanged.

The released GPT-2 adapter scores targets that have already been shifted one
position ahead. A second shift would compare predictions with the wrong
labels. Comparison plots require matching datasets, tokenizers, scoring rules,
and complete evaluations for both models.

## Run the tests

Run these commands from the repository root.

```bash
python -m pip install -e '.[data,eval,dev]'
python -m unittest discover -s tests
ruff check gpt2 tests scripts
ruff format --check gpt2 tests scripts
```

Run the distributed continuation check on suitable CUDA hardware.

```bash
python -m gpt2.verify_gpu_resume --gpus 4
```

The [completed-run analysis](../reports/final/analysis.json) summarizes loss,
gradients, runtime, and evaluation coverage from the training logs. Update
metrics are logged every ten steps.
