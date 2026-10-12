"""Prepare hashed, document-separated token streams for causal pretraining."""

import argparse
import fcntl
import hashlib
import json
import os
import signal
import time
from contextlib import closing, contextmanager
from itertools import islice
from pathlib import Path

import numpy as np

from tokenizer import Tokenizer


@contextmanager
def preparation_lock(directory):
    """Serialize writers to one output directory, including recovery checks."""
    with (directory / "preparation.lock").open("a+b") as handle:
        waited = False
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            waited = True
            print("Waiting for the active preparation writer to finish.", flush=True)
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield waited
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def durable_json(value, path):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def documents(args):
    if args.demo:
        # Synthetic documents exercise preparation and checkpoint integration.
        # They must not be used to report language-model quality.
        for i in range(5000):
            yield f"Example {i}: The student reads a book. The sky is blue. A model predicts the next token."
    elif args.input:
        # The local source format is one UTF-8 document per nonblank line.
        with open(args.input, encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    yield line.strip()
    else:
        import pyarrow.parquet as pq
        from datasets import load_dataset_builder
        from huggingface_hub import HfFileSystem, hf_hub_download

        builder = load_dataset_builder(
            args.hf_dataset,
            name=args.hf_config,
            revision=args.resolved_revision,
        )
        files = builder.config.data_files[args.hf_split]
        filesystem = HfFileSystem()
        for remote in files:
            resolved = filesystem.resolve_path(remote)
            local = hf_hub_download(
                repo_id=args.hf_dataset,
                repo_type="dataset",
                filename=resolved.path_in_repo,
                revision=args.resolved_revision,
            )
            args.consumed_shards.append(resolved.path_in_repo)
            # Cached local shards permit bounded reads and deterministic file
            # lifetimes. Only the current shard needs to be loaded/downloaded.
            with pq.ParquetFile(local) as source:
                for batch in source.iter_batches(
                    batch_size=args.tokenization_batch_size,
                    columns=[args.text_column],
                    use_threads=True,
                ):
                    for text in batch.column(0).to_pylist():
                        if isinstance(text, str) and text.strip():
                            yield text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument("--hf-dataset")
    source.add_argument("--demo", action="store_true")
    parser.add_argument("--hf-config", default=None)
    parser.add_argument("--hf-revision", default="main")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--out", type=Path, default=Path("data/demo"))
    parser.add_argument("--tokenizer", choices=["gpt2", "byte"], default="gpt2")
    parser.add_argument("--val-fraction", type=float, default=0.01)
    parser.add_argument("--max-tokens", type=int, default=10_000_000)
    parser.add_argument("--tokenizer-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--tokenization-batch-size", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-documents", type=int)
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1 or args.max_tokens < 1:
        parser.error("Use 0 < val-fraction < 1 and max-tokens > 0.")
    if args.tokenizer_workers < 1 or args.tokenization_batch_size < 1:
        parser.error("Tokenization workers and batch size must be positive.")
    args.out.mkdir(parents=True, exist_ok=True)
    with preparation_lock(args.out) as waited:
        prepare(args, parser, waited)


def prepare(args, parser, waited=False):
    progress_path = args.out / "preparation.json"
    # A second writer can be launched before the first creates its journal.
    # Inspect state only after acquiring the lock, and reuse the run it waited
    # for. An unrelated fresh invocation must still explicitly request resume.
    resume = args.resume or (waited and progress_path.exists())
    if not resume and any(
        (args.out / f).exists() for f in ("train.bin", "val.bin", "meta.json", "preparation.json")
    ):
        parser.error("Output already has data files; choose a new directory.")
    if resume and not progress_path.exists():
        parser.error("No preparation journal exists in the requested output directory.")
    tokenizer = Tokenizer(args.tokenizer)
    progress = json.loads(progress_path.read_text()) if resume else None
    identity = {
        key: str(getattr(args, key)) if isinstance(getattr(args, key), Path) else getattr(args, key)
        for key in (
            "demo",
            "input",
            "hf_dataset",
            "hf_config",
            "hf_split",
            "text_column",
            "tokenizer",
            "val_fraction",
            "max_tokens",
        )
    }
    if progress and identity != progress["identity"]:
        parser.error(
            "Preparation source, tokenizer, split, or token budget differs from the saved run."
        )
    args.resolved_revision = progress["hf_revision"] if progress else None
    args.consumed_shards = []
    if args.hf_dataset and not progress:
        from huggingface_hub import HfApi

        # Resolve movable branches once, then load from the immutable revision.
        args.resolved_revision = (
            HfApi().dataset_info(args.hf_dataset, revision=args.hf_revision).sha
        )
    count = progress["tokens"] if progress else {"train": 0, "val": 0}
    hashes = {split: hashlib.sha256() for split in count}
    docs = progress["documents"] if progress else 0
    if progress and (args.out / "meta.json").exists():
        meta = json.loads((args.out / "meta.json").read_text())
        for split in count:
            with (args.out / f"{split}.bin").open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    hashes[split].update(chunk)
            if (args.out / f"{split}.bin").stat().st_size != meta["tokens"][split] * 2 or hashes[
                split
            ].hexdigest() != meta["sha256"][split]:
                raise ValueError("Completed preparation failed size/hash verification.")
        print(json.dumps(meta, indent=2))
        return
    stop_requested = {"value": False}

    def request_stop(signum, frame):
        stop_requested["value"] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    started = time.perf_counter()
    mode = "r+b" if progress else "w+b"
    # Larger buffers amortize network-volume writes while save_progress still
    # flushes and fsyncs both streams before committing the document journal.
    with (
        open(args.out / "train.bin", mode, buffering=8 * 1024 * 1024) as train,
        open(args.out / "val.bin", mode, buffering=8 * 1024 * 1024) as val,
    ):
        handles = {"train": train, "val": val}
        if progress:
            for split, handle in handles.items():
                committed_bytes = count[split] * 2
                if (args.out / f"{split}.bin").stat().st_size < committed_bytes:
                    raise ValueError("Prepared token stream is shorter than its saved progress.")
                # Discard only bytes beyond the last durably committed document.
                handle.truncate(committed_bytes)
                handle.seek(0)
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    hashes[split].update(chunk)
                if (
                    progress.get("sha256")
                    and hashes[split].hexdigest() != progress["sha256"][split]
                ):
                    raise ValueError("Prepared token prefix failed checkpoint hash verification.")
                handle.seek(0, 2)

        def save_progress():
            for handle in handles.values():
                handle.flush()
                os.fsync(handle.fileno())
            durable_json(
                {
                    "identity": identity,
                    "hf_revision": args.resolved_revision,
                    "documents": docs,
                    "tokens": count,
                    "sha256": {split: digest.hexdigest() for split, digest in hashes.items()},
                },
                progress_path,
            )

        save_progress()
        with closing(documents(args)) as iterator:
            # Reconstruct the source cursor from the immutable revision. Cached
            # shards are reused; already committed documents are not retokenized.
            for _ in islice(iterator, docs):
                pass
            finished = False
            while not finished:
                texts = list(islice(iterator, args.tokenization_batch_size))
                if not texts:
                    break
                encoded = tokenizer.encode_batch(texts, workers=args.tokenizer_workers)
                for text, ids in zip(texts, encoded):
                    # Assign complete documents before packing. Exact duplicates
                    # remain in one split; near-duplicate filtering is corpus-level.
                    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
                    fraction = int.from_bytes(digest, "big") / 2**64
                    split = "val" if fraction < args.val_fraction else "train"
                    ids.append(tokenizer.eot_id)
                    payload = np.asarray(ids, dtype="<u2").tobytes()
                    handles[split].write(payload)
                    hashes[split].update(payload)
                    count[split] += len(ids)
                    docs += 1
                    if docs % 10000 == 0:
                        save_progress()
                        print(
                            f"documents={docs:,} tokens={sum(count.values()):,} active_seconds={time.perf_counter() - started:.1f}",
                            flush=True,
                        )
                    # Finish the document that crosses the token budget.
                    if sum(count.values()) >= args.max_tokens:
                        finished = True
                        break
                    if stop_requested["value"] or (
                        args.stop_after_documents and docs >= args.stop_after_documents
                    ):
                        stop_requested["value"] = True
                        finished = True
                        break
        save_progress()
    if stop_requested["value"]:
        print(
            f"Saved preparation progress at document {docs:,}; continue with --resume.", flush=True
        )
        raise SystemExit(75)
    if min(count.values()) < 2:
        raise ValueError("Too little data in one split. Increase the token budget/source size.")
    meta = {
        "tokenizer": tokenizer.name,
        "vocab_size": tokenizer.vocab_size,
        "dtype": "<u2",
        "tokens": count,
        "documents": docs,
        "sha256": {s: h.hexdigest() for s, h in hashes.items()},
        "source": args.input.name
        if args.input
        else args.hf_dataset or "synthetic integration corpus",
        "hf_config": args.hf_config,
        "hf_split": args.hf_split,
        "hf_revision": args.resolved_revision,
        "consumed_parquet_shards": args.consumed_shards,
        "tokenizer_workers": args.tokenizer_workers,
        "tokenization_batch_size": args.tokenization_batch_size,
        "val_fraction": args.val_fraction,
        "split_method": "blake2b document hash",
    }
    durable_json(meta, args.out / "meta.json")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
