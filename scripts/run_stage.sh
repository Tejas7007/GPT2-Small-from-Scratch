#!/usr/bin/env bash
# Run one production stage with a constant effective token batch.
# Usage  bash scripts/run_stage.sh preflight|train|resume|evaluate|samples|plot GPU_COUNT
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
if [[ -x .venv/bin/python ]]; then source .venv/bin/activate; fi
mode="${1:-preflight}"
gpu_count="${2:-1}"
case "$gpu_count" in
  1|2|4|8) ;;
  *) printf 'GPU count must be 1, 2, 4, or 8.\n' >&2; exit 2 ;;
esac

data_dir="${TRAIN_DATA_DIR:-$project_root/data/fineweb10b}"
run_dir="${TRAIN_RUN_DIR:-$project_root/runs/gpt2-fineweb10b}"
config_path="${TRAIN_CONFIG:-configs/baseline.json}"
microbatch="${TRAIN_MICROBATCH:-$(python - "$config_path" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))['batch_size'])
PY
)}"
case "$microbatch" in
  1|2|4|8|16|32|64) ;;
  *) printf 'Microbatch must be a power of two from 1 through 64.\n' >&2; exit 2 ;;
esac
accumulation=$((524288 / (1024 * microbatch * gpu_count)))
if ((accumulation < 1)); then
  printf 'Microbatch/GPU count exceeds the configured global batch.\n' >&2
  exit 2
fi

if [[ "$mode" == "plot" ]]; then
  exec python -m gpt2.plot_metrics --run "$run_dir"
fi
if [[ ! -f "$data_dir/meta.json" ]]; then
  printf 'Prepared corpus metadata is missing: %s\n' "$data_dir" >&2
  exit 2
fi
if [[ "$mode" == "evaluate" ]]; then
  exec python -m gpt2.evaluate --checkpoint "$run_dir/best.pt" --data "$data_dir" --out "$run_dir/evaluation.json" --batch-size "${TRAIN_EVAL_BATCH_SIZE:-32}" --verify-data
fi
if [[ "$mode" == "samples" ]]; then
  exec python -m gpt2.sample --checkpoint "$run_dir/best.pt" --out "$run_dir/samples.jsonl"
fi

python - "$gpu_count" <<'PY'
import sys
import torch
requested = int(sys.argv[1])
if not torch.cuda.is_available() or torch.cuda.device_count() < requested:
    raise SystemExit(f'Requires {requested} visible CUDA GPU(s).')
if not torch.cuda.is_bf16_supported():
    raise SystemExit('Baseline configuration requires bf16-capable CUDA hardware.')
for index in range(requested):
    info = torch.cuda.get_device_properties(index)
    print(f'GPU {index}: {info.name}; memory={info.total_memory / 2**30:.1f} GiB')
PY

args=(--config "$config_path" --data "$data_dir" --out "$run_dir" --batch-size "$microbatch" --grad-accum "$accumulation")
# Hold the validation token budget fixed while tuning the microbatch. With
# 10 Mi tokens per evaluation, backend comparisons use equivalent workloads.
validation_batches=$((10485760 / (1024 * microbatch * gpu_count)))
args+=(--eval-batches "$validation_batches")
if [[ -n "${TRAIN_COMPILE:-}" ]]; then
  case "$TRAIN_COMPILE" in
    1) args+=(--compile) ;;
    0) args+=(--no-compile) ;;
    *) printf 'TRAIN_COMPILE must be 0 or 1.\n' >&2; exit 2 ;;
  esac
fi
case "$mode" in
  preflight)
    if [[ -f "$run_dir/last.pt" || -f "$run_dir/metrics.jsonl" ]]; then
      printf 'Preflight output already exists; use train/resume or a fresh TRAIN_RUN_DIR.\n' >&2
      exit 2
    fi
    args+=(--stop-after 20)
    ;;
  train)
    if [[ -f "$run_dir/last.pt" ]]; then args+=(--resume "$run_dir/last.pt"); fi
    ;;
  resume)
    if [[ ! -f "$run_dir/last.pt" ]]; then printf 'No resumable checkpoint found.\n' >&2; exit 2; fi
    args+=(--resume "$run_dir/last.pt")
    ;;
  *) printf 'Mode must be preflight, train, resume, evaluate, samples, or plot.\n' >&2; exit 2 ;;
esac

if ((gpu_count == 1)); then
  exec python -m gpt2.train "${args[@]}"
else
  exec python -m torch.distributed.run --standalone --nproc_per_node="$gpu_count" --module gpt2.train "${args[@]}"
fi


