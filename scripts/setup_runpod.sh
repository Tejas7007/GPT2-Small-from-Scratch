#!/usr/bin/env bash
# Configure a persistent virtual environment using the template CUDA build.
# Record package versions and verify model invariants before any full GPU run.
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

# Keep the project environment on persistent storage while inheriting the
# template's CUDA-enabled torch build. Restarting a container must not erase
# the experiment's Python dependency installation.
if [[ ! -x .venv/bin/python ]]; then
  python -m venv --system-site-packages .venv
fi
source .venv/bin/activate

# Reuse the CUDA-compatible PyTorch build supplied by the pod template.
# Installing a second torch distribution can silently change CUDA support.
python - <<'PY'
import torch
major, minor = (int(part) for part in torch.__version__.split('.')[:2])
if (major, minor) < (2, 3):
    raise SystemExit('A PyTorch >=2.3 template is required.')
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
PY

python -m pip install 'numpy>=1.26,<3' 'tiktoken>=0.7,<1' 'matplotlib>=3.8,<4' 'datasets>=2.19,<5'
mkdir -p environment
python -m pip list --format=json > environment/packages.json
python - <<'PY'
import json
from pathlib import Path
packages = json.loads(Path('environment/packages.json').read_text())
lock = '\n'.join(f"{row['name']}=={row['version']}" for row in packages) + '\n'
Path('environment/requirements.lock').write_text(lock)
PY
python -m unittest discover -s tests -p test_model.py

