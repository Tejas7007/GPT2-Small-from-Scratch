"""Render training/validation loss and optimization diagnostics from run logs."""

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def matching_hellaswag_reports(model, reference):
    """Compare only complete, identical validation corpora and scoring protocols."""
    from evaluate_hellaswag import DATASET, DATASET_REVISION, EVALUATION_PROTOCOL, VALIDATION_ROWS

    expected = {
        "task": "hellaswag",
        "split": "validation",
        "fewshot": 0,
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "evaluated_examples": VALIDATION_ROWS,
        "available_examples": VALIDATION_ROWS,
        "complete_validation_sweep": True,
        "tokenizer": "gpt2",
        "context_length": 1024,
        "chance_accuracy": 0.25,
        "evaluation_protocol": EVALUATION_PROTOCOL,
    }
    if any(
        report.get(key) != value for report in (model, reference) for key, value in expected.items()
    ):
        return False
    digest = model.get("validation_rows_sha256")
    return (
        isinstance(digest, str)
        and len(digest) == 64
        and digest == reference.get("validation_rows_sha256")
        and model.get("precision") in ("bfloat16", "float32")
        and model.get("precision") == reference.get("precision")
        and reference.get("checkpoint") == "openai-community/gpt2"
    )


def plot_hellaswag_quality(run, out):
    """Optional downstream comparison; unavailable or failed sweeps cannot block plots."""
    try:
        status_path = run / "hellaswag_status.json"
        if (
            status_path.is_file()
            and json.loads(status_path.read_text()).get("status") != "complete"
        ):
            return False
        model = json.loads((run / "hellaswag.json").read_text())
        reference = json.loads((run / "hellaswag_reference.json").read_text())
        if not matching_hellaswag_reports(model, reference):
            print("Optional quality comparison omitted: validation corpus or protocol differs.")
            return False
        metrics = ("accuracy", "accuracy_token_normalized", "accuracy_character_normalized")
        scores = np.array([[report[key] for key in metrics] for report in (model, reference)])
        if not np.isfinite(scores).all() or not ((0 <= scores) & (scores <= 1)).all():
            raise ValueError("HellaSwag accuracies must be finite values between zero and one.")
        figure, axis = plt.subplots(figsize=(8, 4.8))
        x = np.arange(3)
        width = 0.34
        for positions, values, color, label in (
            (x - width / 2, scores[0], "#245B9B", "Our trained model"),
            (x + width / 2, scores[1], "#C75A31", "Released GPT-2"),
        ):
            bars = axis.bar(positions, values, width, color=color, label=label)
            axis.bar_label(bars, labels=[f"{value:.1%}" for value in values], padding=3, fontsize=9)
        axis.axhline(0.25, color="#666666", linestyle="--", linewidth=1, label="Chance · 25%")
        axis.set_xticks(x, ["Raw ending likelihood", "Per ending token", "Per ending character"])
        axis.set(
            ylabel="Fraction of correct answers",
            ylim=(0, 1),
            title="Zero-shot HellaSwag · all 10,042 validation questions",
        )
        axis.legend(frameon=False)
        figure.tight_layout()
        save_figure(figure, out, "quality")
        return True
    except FileNotFoundError:
        return False
    except (OSError, ValueError, TypeError, KeyError, ImportError) as error:
        print(f"Optional quality comparison omitted: {type(error).__name__}: {error}")
        return False


def save_figure(figure, directory, stem):
    for extension in ("png", "svg"):
        temporary = directory / f"{stem}.tmp.{extension}"
        figure.savefig(temporary, dpi=200, bbox_inches="tight")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(directory / f"{stem}.{extension}")
    plt.close(figure)


def plot_convergence(train, val, tokens_per_step, reference, out):
    """Show later convergence without the initial loss setting the y-axis range."""

    def tokens(row):
        return row.get("tokens_seen", row["step"] * tokens_per_step)

    late_train = [row for row in train if tokens(row) >= 1e9 and np.isfinite(row["loss"])]
    late_val = [row for row in val if tokens(row) >= 1e9 and np.isfinite(row["loss"])]
    if len({tokens(row) for row in late_train}) < 2:
        return False
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.plot(
        [tokens(row) / 1e9 for row in late_train],
        [row["loss"] for row in late_train],
        color="#245B9B",
        linewidth=1.5,
        label="Training batch loss",
    )
    if late_val:
        axis.plot(
            [tokens(row) / 1e9 for row in late_val],
            [row["loss"] for row in late_val],
            color="#C75A31",
            linewidth=1.8,
            marker="o",
            markersize=3.5,
            label="Fixed-window validation loss",
        )
    values = [row["loss"] for row in late_train + late_val]
    low, high = min(values), max(values)
    padding = max((high - low) * 0.06, 0.02)
    low, high = low - padding, high + padding
    if reference and reference.get("complete_window_sweep") and np.isfinite(reference["loss"]):
        loss = reference["loss"]
        axis.axhline(
            loss,
            color="#666666",
            linestyle="--",
            linewidth=1.2,
            label=f"Released GPT-2 full held-out sweep · {loss:.3f}",
        )
        if not low <= loss <= high:
            axis.text(
                0.98,
                0.02,
                f"Reference loss {loss:.3f} is outside the displayed range",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                color="#666666",
            )
    # Bound the y-axis using only the displayed train/validation measurements.
    axis.set(
        xlabel="Processed training tokens (billions)",
        ylabel="Cross-entropy (nats/token)",
        title="Convergence after one billion training tokens",
        ylim=(low, high),
    )
    axis.legend(frameon=False)
    figure.tight_layout()
    save_figure(figure, out, "convergence")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or args.run / "figures"
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.run / "manifest.json").read_text())
    records = [
        json.loads(line) for line in (args.run / "metrics.jsonl").read_text().splitlines() if line
    ]
    train = sorted((row for row in records if row["kind"] == "train"), key=lambda row: row["step"])
    val = sorted((row for row in records if row["kind"] == "val"), key=lambda row: row["step"])
    if not train:
        raise ValueError("The run has no recorded training updates.")
    per_step = manifest["tokens_per_step"]

    def count(row):
        return row.get("tokens_seen", row["step"] * per_step)

    maximum = max(count(row) for row in train)
    scale, unit = (
        (1e9, "billions")
        if maximum >= 1e9
        else (1e6, "millions")
        if maximum >= 1e6
        else (1e3, "thousands")
    )
    x = np.array([count(row) / scale for row in train])
    vx = np.array([count(row) / scale for row in val])
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "svg.fonttype": "none",
        }
    )
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.plot(
        x,
        [row["loss"] for row in train],
        color="#245B9B",
        linewidth=1.5,
        label="Training batch loss",
    )
    if val:
        axis.plot(
            vx,
            [row["loss"] for row in val],
            color="#C75A31",
            linewidth=1.8,
            marker="o",
            markersize=3.5,
            label="Fixed-window validation loss",
        )
    reference_path = args.run / "reference_evaluation.json"
    reference = None
    if reference_path.exists():
        reference = json.loads(reference_path.read_text())
        axis.axhline(
            reference["loss"],
            color="#666666",
            linestyle="--",
            linewidth=1.2,
            label="Released GPT-2 · complete held-out evaluation",
        )
    axis.set(
        xlabel=f"Processed training tokens ({unit})",
        ylabel="Cross-entropy (nats/token)",
        title=f"Causal pretraining · {manifest['parameters'] / 1e6:.2f}M parameters",
    )
    axis.legend(frameon=False)
    figure.tight_layout()
    save_figure(figure, out, "loss")
    plot_convergence(train, val, per_step, reference, out)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(x, [row["lr"] for row in train], color="#245B9B", linewidth=1.6)
    axes[0].set(
        xlabel=f"Processed tokens ({unit})", ylabel="Learning rate", title="Optimizer schedule"
    )
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[1].plot(
        [row["step"] for row in train],
        [row["tokens_per_second"] / 1000 for row in train],
        color="#37785A",
        linewidth=1.4,
    )
    axes[1].set(
        xlabel="Optimizer update",
        ylabel="Thousands of tokens / second",
        title="Measured update throughput",
    )
    figure.tight_layout()
    save_figure(figure, out, "optimization")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    if val:
        axes[0].plot(
            vx, [np.exp(row["loss"]) for row in val], color="#C75A31", marker="o", markersize=3
        )
        axes[0].set_yscale("log")
    axes[0].set(
        xlabel=f"Processed tokens ({unit})",
        ylabel="Validation perplexity",
        title="Held-out prediction uncertainty",
    )
    axes[1].plot(x, [row["grad_norm"] for row in train], color="#76589B", linewidth=1.4)
    axes[1].axhline(1.0, color="#666666", linestyle="--", linewidth=1, label="Clipping threshold")
    axes[1].set(
        xlabel=f"Processed tokens ({unit})",
        ylabel="Gradient norm before clipping",
        title="Optimization stability",
    )
    axes[1].legend(frameon=False)
    figure.tight_layout()
    save_figure(figure, out, "diagnostics")

    elapsed = [row.get("elapsed_seconds") for row in train]
    if all(value is not None for value in elapsed):
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        hours = np.array(elapsed) / 3600
        axes[0].plot(hours, [row["loss"] for row in train], color="#245B9B", linewidth=1.4)
        axes[0].set(
            xlabel="Active trainer time (hours)",
            ylabel="Training cross-entropy",
            title="Loss over measured run time",
        )
        memory = [row.get("peak_cuda_memory_bytes") for row in train]
        if all(value is not None for value in memory):
            axes[1].plot(x, np.array(memory) / 2**30, color="#37785A", linewidth=1.4)
            axes[1].set(
                xlabel=f"Processed tokens ({unit})",
                ylabel="Peak allocated CUDA memory (GiB)",
                title="Device memory",
            )
        else:
            axes[1].plot(hours, x, color="#37785A", linewidth=1.4)
            axes[1].set(
                xlabel="Active trainer time (hours)",
                ylabel=f"Processed tokens ({unit})",
                title="Measured training progress",
            )
        figure.tight_layout()
        save_figure(figure, out, "runtime")
    plot_hellaswag_quality(args.run, out)
    # Exclude the first logged update when subsequent observations exist;
    # reported medians are diagnostic rather than end-to-end run timings.
    speeds = [row["tokens_per_second"] for row in train[1:]] or [train[0]["tokens_per_second"]]
    sessions_path = args.run / "sessions.jsonl"
    sessions = (
        [json.loads(line) for line in sessions_path.read_text().splitlines() if line]
        if sessions_path.exists()
        else []
    )
    report = {
        "parameters": manifest["parameters"],
        "last_step": train[-1]["step"],
        "training_tokens": count(train[-1]),
        "last_train_loss": train[-1]["loss"],
        "last_sampled_val_loss": val[-1]["loss"] if val else None,
        "best_sampled_val_loss": min(row["loss"] for row in val) if val else None,
        "median_logged_update_tokens_per_second": float(np.median(speeds)),
        "hardware": manifest["hardware"],
        "precision": manifest["precision"],
        "world_size": manifest["world_size"],
        "global_token_batch": per_step,
        "curve_smoothing": "none",
        "throughput_scope": "logged updates, excluding validation and checkpoint time",
        "completed_session_wall_seconds": sum(
            row.get("wall_seconds", 0.0) for row in sessions if row.get("event") == "end"
        ),
    }
    (out / "run_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
