#!/usr/bin/env bash
# Execute the four-H100 recipe after resume checks and a measured benchmark.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
# Use the recorded dataset revision unless PREP_HF_REVISION is set.
prep_revision="${PREP_HF_REVISION:-9bb295ddab0e05d785b879661af7260fed5140fc}"
cache_root="${TRAIN_CACHE_DIR:-$project_root/cache}"
export HF_HOME="${HF_HOME:-$cache_root/huggingface}"
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-$cache_root/tiktoken}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$cache_root/torchinductor}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$cache_root/pip}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TRAIN_CONFIG="${TRAIN_CONFIG:-configs/h100-4gpu.json}"
export TRAIN_RUN_DIR="${TRAIN_RUN_DIR:-$project_root/runs/gpt2-fineweb10b-h100-4gpu}"
export TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-$project_root/data/fineweb10b}"
export TRAIN_EVALUATE_REFERENCE="${TRAIN_EVALUATE_REFERENCE:-1}"
mkdir -p logs environment
bash scripts/setup_runpod.sh
source .venv/bin/activate
python -m pip install -r requirements-eval.txt
if [[ ! -f runs/gpu-resume-verification/passed.txt ]]; then python -m gpt2.verify_gpu_resume --gpus 4; fi
if [[ ! -f data/calibration/meta.json ]]; then
  python -m gpt2.prepare_data --hf-dataset HuggingFaceFW/fineweb --hf-config sample-10BT --hf-revision "$prep_revision" --out data/calibration --max-tokens 2000000 --tokenizer-workers "${PREP_TOKENIZER_WORKERS:-32}" --tokenization-batch-size 4096
fi
if [[ ! -f runs/benchmarks/h100-4gpu/benchmark.json ]]; then
  benchmark_args=(--config "$TRAIN_CONFIG" --data data/calibration --gpus 4 --microbatches 32 64 --out runs/benchmarks/h100-4gpu)
  python -m gpt2.benchmark "${benchmark_args[@]}"
fi
python - <<'PY'
import json
from pathlib import Path
report=json.loads(Path('runs/benchmarks/h100-4gpu/benchmark.json').read_text())
if not report['gpu_performance_measured']:
 raise SystemExit('GPU benchmark did not establish a usable training configuration.')
PY
source runs/benchmarks/h100-4gpu/launch.env
if [[ ! -f "$TRAIN_DATA_DIR/meta.json" ]]; then
  args=(--hf-dataset HuggingFaceFW/fineweb --hf-config sample-10BT --hf-revision "$prep_revision" --out "$TRAIN_DATA_DIR" --max-tokens 10000000000 --tokenizer-workers "${PREP_TOKENIZER_WORKERS:-32}" --tokenization-batch-size 4096)
  if [[ -f "$TRAIN_DATA_DIR/preparation.json" ]]; then args+=(--resume); fi
  python -m gpt2.prepare_data "${args[@]}"
fi
if [[ ! -f "$TRAIN_RUN_DIR/last.pt" ]]; then
  bash scripts/run_stage.sh preflight 4
fi
bash scripts/run_stage.sh train 4
bash scripts/complete_run.sh "$TRAIN_RUN_DIR"
