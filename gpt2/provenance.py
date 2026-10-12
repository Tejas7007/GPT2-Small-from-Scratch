"""Record source hashes, revisions, and software versions for reproducibility."""

import platform
import subprocess
from pathlib import Path

import torch

from gpt2.checkpoint import file_sha256


def source_files(root):
    """Enumerate executable source and its configuration using repository paths."""
    return sorted(
        path
        for path in {
            *root.glob("*.py"),
            *root.glob("gpt2/*.py"),
            *root.glob("scripts/*.sh"),
            *root.glob("configs/*.json"),
            *root.glob("requirements*.txt"),
            root / "pyproject.toml",
        }
        if path.is_file()
    )


def provenance():
    """Record source identity and the Python/PyTorch versions used for training."""
    root = Path(__file__).resolve().parents[1]
    source_hashes = {
        path.relative_to(root).as_posix(): file_sha256(path) for path in source_files(root)
    }
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True))
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit, dirty = None, None
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "source_sha256": source_hashes,
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
    }
