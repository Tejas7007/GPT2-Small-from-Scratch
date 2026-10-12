#!/usr/bin/env bash
# Prepare document-separated FineWeb tokens in an independently resumable job.
# A data-only environment allows tokenization to run before GPU provisioning.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
# Use the recorded dataset revision unless PREP_HF_REVISION is set.
prep_revision="${PREP_HF_REVISION:-9bb295ddab0e05d785b879661af7260fed5140fc}"
cache_root="${TRAIN_CACHE_DIR:-$project_root/cache}"
export HF_HOME="${HF_HOME:-$cache_root/huggingface}"
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-$cache_root/tiktoken}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$cache_root/pip}"
mkdir -p "$HF_HOME" "$TIKTOKEN_CACHE_DIR" "$PIP_CACHE_DIR" environment logs
if [[ ! -x .venv-data/bin/python ]]; then python3 -m venv --system-site-packages .venv-data; fi
.venv-data/bin/python -m pip install 'numpy>=1.26,<3' 'tiktoken>=0.7,<1' 'datasets>=2.19,<5'
.venv-data/bin/python -m pip list --format=json > environment/data-packages.json
args=(--hf-dataset HuggingFaceFW/fineweb --hf-config sample-10BT --hf-revision "$prep_revision" --out data/fineweb10b --max-tokens 10000000000 --tokenizer-workers "${PREP_TOKENIZER_WORKERS:-32}" --tokenization-batch-size 2048)
if [[ -f data/fineweb10b/preparation.json ]]; then args+=(--resume); fi
exec .venv-data/bin/python -m gpt2.prepare_data "${args[@]}"
