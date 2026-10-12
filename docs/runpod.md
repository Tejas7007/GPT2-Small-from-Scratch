# GPU setup and training

The recorded experiment used one node with four H100 80GB GPUs. Training needs
a compatible CUDA PyTorch environment and enough storage for the corpus and
checkpoints. Place these files on persistent storage so they survive a
container restart.

## Environment and data

Run commands from the repository root. The setup script creates a virtual
environment using the CUDA-enabled PyTorch supplied by the pod template.
Cache files default to the repository's `cache` directory. `TRAIN_CACHE_DIR`
can select another location.

```bash
bash scripts/setup_runpod.sh
source .venv/bin/activate
bash scripts/prepare_fineweb.sh
```

FineWeb preparation runs on CPUs and can finish before GPU training starts.
Allow approximately 20GB for token files, plus space for downloaded source
shards, package caches, multiple 1.5GB full checkpoints, and milestone weights.
The preparation wrapper uses the dataset revision recorded by this experiment.
`PREP_HF_REVISION` selects another revision for a new experiment.
[The training guide](training.md) explains the document split and how interrupted
preparation resumes.

## Check the training profile

First test distributed checkpoint continuation, then measure the candidate
microbatches.

```bash
python -m gpt2.verify_gpu_resume --gpus 4
python -m gpt2.benchmark \
  --config configs/h100-4gpu.json \
  --data data/fineweb10b \
  --gpus 4 \
  --microbatches 32 64 \
  --out runs/benchmarks/h100-4gpu
```

Use a fresh output directory for each benchmark. Check the loss, GPU memory,
checkpoint duration, and update timings before starting the full training run.
[The runtime guide](benchmark.md) explains these measurements and the projected
training duration.

## Train and evaluate

```bash
export TRAIN_CONFIG=configs/h100-4gpu.json
export TRAIN_RUN_DIR=runs/gpt2-fineweb10b-h100-4gpu
export TRAIN_DATA_DIR=data/fineweb10b
bash scripts/run_stage.sh preflight 4
bash scripts/run_stage.sh train 4
bash scripts/run_stage.sh evaluate 4
bash scripts/run_stage.sh samples 4
bash scripts/run_stage.sh plot 4
```

The preflight performs twenty updates on the full learning-rate schedule and
saves a checkpoint. The training stage continues from that checkpoint when it
exists. Use the `resume` stage to require an existing checkpoint explicitly.

Continuation uses the original GPU count, microbatch, accumulation, dataset,
and schedule. These settings determine the next batch and optimizer update,
so changing them would create a different training trajectory. The checkpoint
stores model weights, optimizer state, random state, and each worker's sampling
position.

After training, the evaluation stage scores the held-out corpus using the
selected checkpoint. The sample stage generates the five fixed-prompt
continuations, and the plot stage draws the loss and runtime figures. Refer to
[the evaluation guide](evaluation.md) for the released-model comparison and
HellaSwag commands.

To evaluate an existing checkpoint, create the prompt suite, and draw the
figures in one command, use the helper with its run directory.

```bash
bash scripts/complete_run.sh runs/gpt2-fineweb10b-h100-4gpu
```

It uses `best.pt` when available and otherwise uses `last.pt`.
`TRAIN_CHECKPOINT` selects another saved model, and `TRAIN_EVAL_DEVICE`
selects the inference device. Results are written into the run directory.
