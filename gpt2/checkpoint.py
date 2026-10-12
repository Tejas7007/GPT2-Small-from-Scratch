"""Durable checkpoint replacement and streamed file identity verification.

A successful return guarantees that the completed archive and its directory
entry were flushed to storage. Failures before replacement leave the previous
target intact. Only callers that own a path may invoke replacement concurrently.
"""

import hashlib
import os
import shutil
import tempfile

import torch


def atomic_save(payload, path):
    """Serialize to staging and replace the durable target only after a full copy.

    Staging isolates small serializer writes from slow attached volumes. The
    temporary target remains beside the final path so rename is atomic. File
    fsync precedes rename, and directory fsync commits the replacement entry.
    A directory-fsync error occurs after replacement, when the complete new
    archive is already visible but its crash durability could not be confirmed.
    """
    temporary = path.with_suffix(".tmp")
    try:
        # Torch's archive serializer performs many small writes. Stage on the
        # local storage, then transfer large blocks to the persistent volume.
        # The previous checkpoint remains intact until the durable replacement.
        with tempfile.TemporaryFile(dir=os.environ.get("TRAIN_CHECKPOINT_STAGE_DIR")) as staged:
            torch.save(payload, staged)
            staged.flush()
            staged.seek(0)
            with temporary.open("wb") as handle:
                shutil.copyfileobj(staged, handle, length=8 * 1024 * 1024)
                handle.flush()
                os.fsync(handle.fileno())
        temporary.replace(path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def file_sha256(path):
    """Hash arbitrary-sized files with bounded host-memory use."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
