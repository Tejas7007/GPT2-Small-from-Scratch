# Project walkthrough

Start with one prediction, then follow how it becomes a training update.
The main path runs through `model.py`, `data.py`, and `train.py`. Once those
fit together, the preparation, checkpoint, and evaluation tools have a clear
place in the project.

## Read one forward pass

Begin with `GPT.forward` in [model.py](../gpt2/model.py). It takes a batch of
token IDs, adds token and position embeddings, passes the vectors through
twelve transformer blocks, and projects the result into vocabulary scores.
The [architecture guide](architecture.md) walks through the shapes and the
attention equation alongside the diagram.

Inside a block, attention combines information from the current position
and its earlier context. The feedforward network then transforms the features
at each position. Both operations add their output to the incoming vectors.
Following these two additions is a useful way to understand the residual
paths in the code.

The manual attention branch shows the query/key product, causal mask, softmax,
and weighted value sum directly. The SDPA branch performs that calculation
through PyTorch's attention interface. In the
[model tests](../tests/test_model.py), changing a later input leaves earlier
predictions unchanged, and the two attention implementations are compared
through both forward values and backward gradients.

## See where the inputs and targets come from

[tokenizer.py](../gpt2/tokenizer.py) supplies the GPT-2 BPE tokenizer used by
the experiment. It also has a byte-level mode for small test fixtures.
[prepare_data.py](../gpt2/prepare_data.py) reads FineWeb documents, assigns
whole documents to training or validation, tokenizes them, and appends an
end-of-text separator. The resulting token files feed the training and
validation loaders.

Preparation can continue after an interruption, starting from the last
completed document. The [training guide](training.md) gives the dataset
revision, split method, corpus counts, and preparation command.

Next, read `TokenStream.batch` in [data.py](../gpt2/data.py). It reads a source
window of 1,025 tokens and splits it into 1,024 inputs and 1,024 targets,
offset by one position. That small shift defines the prediction task. The
loss at every input position measures the probability assigned to the next
source token.

During training, the loader shuffles window indices once per epoch. Every
GPU process builds the same permutation and takes a different slice, so
the four GPUs work on different windows. Its saved epoch and cursor describe
where the next batch should begin when the run continues.

## Follow an optimizer update

In [train.py](../gpt2/train.py), `main` creates the model, optimizer, streams,
and distributed process group. Each process runs on one GPU. The full run
uses sixty-four sequences per GPU and two microbatches per update, producing
a global batch of 524,288 tokens.

Follow the loop from `optimizer.zero_grad` to `scaler.step(optimizer)`. Each
microbatch runs a forward pass and backpropagates half of its mean loss.
The first microbatch accumulates gradients locally; the second synchronizes
them through DistributedDataParallel. Averaging across ranks gives the
gradient of the full batch's mean loss.

Before AdamW takes the step, the trainer measures the gradient norm and clips
it to 1.0. It then updates the parameters and writes a metric record when
the logging interval is reached. The record includes the batch loss,
learning rate, gradient norm, update duration, throughput, and peak CUDA
allocation. Reading `learning_rate` alongside the loop shows how warmup and
cosine decay use the absolute update count.

## Resume training

The optimizer's moving averages and the loader's position both affect the
next update. A training checkpoint therefore includes those states along
with model weights, random states, and the precision scaler. The resume
branch in `train.main` restores them before entering the loop.

[checkpoint.py](../gpt2/checkpoint.py) handles the temporary write, flush,
and file replacement. Signals are handled at a complete update boundary,
giving the save routine a fully updated model and optimizer to write.
[verify_gpu_resume.py](../gpt2/verify_gpu_resume.py) compares continued
training with interrupted and resumed training, while the CPU workflow
tests exercise the same sequence on a small model.

## Read the model's results

During training, validation repeatedly scores the same seeded sample of
held-out windows. The standalone [evaluate.py](../gpt2/evaluate.py) later
scores all complete, non-overlapping validation windows. It adds their
negative log likelihoods and divides by the number of target tokens to
produce the final mean loss and perplexity.

[evaluate_hellaswag.py](../gpt2/evaluate_hellaswag.py) asks a different question.
For each context, it scores four possible endings and chooses the one with
the lowest ending loss. Prompt and padding positions are excluded from that
loss. The script reports raw, token-normalized, and character-normalized
accuracy. Both evaluators can load the released GPT-2 model, so the reference
comparison uses the same data and calculation. The
[evaluation guide](evaluation.md) explains the protocols and score interpretation.

For text generation, [generate.py](../gpt2/generate.py) prints a continuation
for a supplied prompt. [sample.py](../gpt2/sample.py) runs the ordered prompt
suite and writes each prompt, continuation, seed, and sampling settings to
JSONL. Reading all five continuations shows both the model's learned text
structure and its repetition, drift, and factual errors.

## Move from records to figures

[plot_metrics.py](../gpt2/plot_metrics.py) reads the event log and draws loss,
learning-rate, gradient, memory, and runtime plots.
[analyze_results.py](../scripts/analyze_results.py) produces the milestone
table and model/reference comparisons with uncertainty intervals from the
saved metrics.
[render_visuals.py](../scripts/render_visuals.py) builds the documentation
diagrams and schematic animation from the report. Its
[generation guide](visuals.md) gives the local command and font requirements.

## Repository map

| Location | Contents |
| --- | --- |
| `gpt2/` | Model, data loader, trainer, and evaluators |
| `configs/` | Training profiles and the ordered generation prompt suite |
| `scripts/` | Setup, execution stages, analysis, and documentation rendering |
| `tests/` | Model, sampler, resume, evaluator, and workflow tests |
| `reports/` | Metric logs, evaluation results, and measured figures |
| `docs/` | Architecture, training, evaluation, and walkthrough guides |

## Execution and reporting tools

| Script | Role |
| --- | --- |
| `setup_runpod.sh` | Create the Python environment and install dependencies |
| `prepare_fineweb.sh` | Prepare or continue the FineWeb token files |
| `run_stage.sh` | Run preflight, training, resume, evaluation, sampling, or plotting |
| `complete_run.sh` | Evaluate an existing checkpoint and create samples and plots |
| `run_experiment.sh` | Prepare data, benchmark the trainer, and run the selected configuration |
| `analyze_results.py` | Summarize training milestones and draw model/reference comparisons |
| `render_visuals.py` | Generate documentation SVGs and the training animation locally |
