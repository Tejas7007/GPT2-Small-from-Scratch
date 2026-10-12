"""Verify distributed CUDA checkpoint continuation before full pretraining."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("runs/gpu-resume-verification"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    data = args.out / "data"
    environment = {**os.environ, "OMP_NUM_THREADS": "1"}
    subprocess.run(
        [
            sys.executable,
            "-m",
            "gpt2.prepare_data",
            "--demo",
            "--tokenizer",
            "byte",
            "--out",
            str(data),
            "--max-tokens",
            "100000",
        ],
        env=environment,
        check=True,
    )
    common = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={args.gpus}",
        "--module",
        "gpt2.train",
        "--config",
        "configs/integration.json",
        "--data",
        str(data),
        "--device",
        "cuda",
        "--precision",
        "bf16",
        "--tokens",
        str(2048 * args.gpus),
    ]
    full, resumed = args.out / "full", args.out / "resumed"
    for extra in (
        ["--out", str(full)],
        ["--out", str(resumed), "--stop-after", "4"],
        ["--out", str(resumed), "--resume", str(resumed / "last.pt")],
    ):
        subprocess.run(common + extra, env=environment, check=True)
    a = torch.load(full / "last.pt", weights_only=True)
    b = torch.load(resumed / "last.pt", weights_only=True)
    if a["step"] != 8 or a["step"] != b["step"] or a["tokens_seen"] != b["tokens_seen"]:
        raise AssertionError("Checkpoint step/token counts differ.")
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=1e-5, atol=1e-6)
    for key, state in a["optimizer"]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(
                value, b["optimizer"]["state"][key][name], rtol=1e-5, atol=1e-6
            )
    for index in range(args.gpus):
        for name in ("train_state", "val_state"):
            for key, value in a["rank_states"][index][name].items():
                other = b["rank_states"][index][name][key]
                if isinstance(value, torch.Tensor):
                    if not torch.equal(value, other):
                        raise AssertionError("Sampler state differs.")
                elif value != other:
                    raise AssertionError("Sampler cursor/configuration differs.")
    (args.out / "passed.txt").write_text(
        "Distributed bf16 checkpoint, optimizer, and sampler continuation verified.\n"
    )
    print("CUDA checkpoint continuation verified.", flush=True)


if __name__ == "__main__":
    main()
