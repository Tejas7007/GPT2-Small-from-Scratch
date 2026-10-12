#!/usr/bin/env bash
# Evaluate an existing checkpoint, generate the prompt suite, and draw figures.
# Usage  bash scripts/complete_run.sh RUN_DIRECTORY
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
if [[ -x .venv/bin/python ]]; then source .venv/bin/activate; fi
run_dir="${1:-${TRAIN_RUN_DIR:-$project_root/runs/gpt2-fineweb10b}}"
data_dir="${TRAIN_DATA_DIR:-$project_root/data/fineweb10b}"
checkpoint="${TRAIN_CHECKPOINT:-$run_dir/best.pt}"
device="${TRAIN_EVAL_DEVICE:-auto}"
if [[ -z "${TRAIN_CHECKPOINT:-}" && ! -f "$checkpoint" ]]; then
  checkpoint="$run_dir/last.pt"
fi
if [[ ! -f "$checkpoint" ]]; then
  printf 'Checkpoint missing %s\n' "$checkpoint" >&2
  exit 2
fi

python -m gpt2.evaluate \
  --checkpoint "$checkpoint" \
  --data "$data_dir" \
  --out "$run_dir/evaluation.json" \
  --batch-size "${TRAIN_EVAL_BATCH_SIZE:-32}" \
  --device "$device" \
  --verify-data
python -m gpt2.sample \
  --checkpoint "$checkpoint" \
  --out "$run_dir/samples.jsonl" \
  --device "$device"

if [[ "${TRAIN_EVALUATE_REFERENCE:-0}" == "1" ]]; then
  python -m gpt2.evaluate \
    --reference gpt2 \
    --data "$data_dir" \
    --out "$run_dir/reference_evaluation.json" \
    --batch-size "${TRAIN_EVAL_BATCH_SIZE:-32}" \
    --device "$device" \
    --verify-data
fi

# A failed optional benchmark still allows the loss and runtime plots to finish.
hellaswag_status="skipped"
hellaswag_exit=0
if [[ "${TRAIN_EVALUATE_HELLASWAG:-1}" == "1" ]]; then
  if python -m gpt2.evaluate_hellaswag \
    --checkpoint "$checkpoint" \
    --out "$run_dir/hellaswag.json" \
    --batch-size "${TRAIN_HELLASWAG_BATCH_SIZE:-8}" \
    --device "$device"; then
    hellaswag_status="complete"
  else
    hellaswag_exit=$?
    hellaswag_status="failed"
    printf 'HellaSwag evaluation failed with exit %s\n' "$hellaswag_exit" >&2
  fi
fi

# Record the current outcome so plots do not reuse a previous benchmark score.
python - "$run_dir" "$hellaswag_status" "$hellaswag_exit" <<'STATUS'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]) / "hellaswag_status.json"
temporary = path.with_suffix(".tmp")
temporary.write_text(
    json.dumps({"status": sys.argv[2], "exit_code": int(sys.argv[3])}, indent=2) + "\n"
)
temporary.replace(path)
STATUS

python -m gpt2.plot_metrics --run "$run_dir"
