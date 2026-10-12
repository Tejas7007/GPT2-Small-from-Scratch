#!/usr/bin/env python3
"""Summarize training and evaluation metrics and draw comparison figures.

The script reads the report's JSON and JSONL files, checks the scores and token
counts, and writes a milestone table and model comparison plots.

Example
    python scripts/analyze_results.py --report reports/final
"""

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import zipfile
from collections import Counter
from pathlib import Path


def read_json(path):
    """Read a report object, allowing its absence only at the caller's discretion."""
    return json.loads(path.read_text(encoding="utf-8"))


def read_records(path):
    """Parse JSON objects from nonempty log lines."""
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path.name} line {number} is invalid JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path.name} line {number} is not an object")
        records.append(row)
    return records


def sha256(path):
    """Identify the source bytes without retaining machine-specific paths."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def retained_source_integrity(path, expected_sources):
    """Verify exact ZIP membership and the bytes of every retained source file."""
    details = {
        "expected_files": len(expected_sources),
        "archive_files": 0,
        "missing_files": [],
        "unlisted_files": [],
        "duplicate_files": [],
        "mismatched_files": [],
        "archive_error": None,
    }
    try:
        with zipfile.ZipFile(path) as archive:
            entries = [entry for entry in archive.infolist() if not entry.is_dir()]
            counts = Counter(entry.filename for entry in entries)
            details["archive_files"] = len(entries)
            details["missing_files"] = sorted(set(expected_sources) - set(counts))
            details["unlisted_files"] = sorted(set(counts) - set(expected_sources))
            details["duplicate_files"] = sorted(name for name, count in counts.items() if count > 1)
            for entry in entries:
                name = entry.filename
                if name not in expected_sources or counts[name] != 1:
                    continue
                digest = hashlib.sha256()
                with archive.open(entry) as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest() != expected_sources[name]:
                    details["mismatched_files"].append(name)
    except (OSError, zipfile.BadZipFile, RuntimeError, NotImplementedError) as error:
        details["archive_error"] = type(error).__name__
    passed = bool(expected_sources) and not any(
        details[key]
        for key in (
            "missing_files",
            "unlisted_files",
            "duplicate_files",
            "mismatched_files",
            "archive_error",
        )
    )
    return passed, details


def quantile(values, fraction):
    """Use linear interpolation, including small measured sample collections."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values):
    """Summarize only the supplied observations without extrapolating missing steps."""
    if not values:
        return None
    return {
        "observations": len(values),
        "minimum": min(values),
        "median": statistics.median(values),
        "p05": quantile(values, 0.05),
        "p95": quantile(values, 0.95),
        "maximum": max(values),
    }


def wilson_interval(correct, total):
    """Compute a two-sided binomial Wilson interval with 95 percent coverage.

    This quantifies question-level accuracy uncertainty under a binomial model.
    It is not variation across training seeds or a paired model-comparison test.
    """
    if total < 1 or not 0 <= correct <= total:
        raise ValueError("Wilson counts must satisfy 0 <= correct <= total and total > 0")
    z = 1.959963984540054
    proportion = correct / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return {"lower": max(0.0, center - radius), "upper": min(1.0, center + radius)}


def finite_issues(value, prefix):
    """Inspect nested numerical fields, including records not drawn in figures."""
    if isinstance(value, float) and not math.isfinite(value):
        return [prefix]
    if isinstance(value, dict):
        return [
            issue for key, item in value.items() for issue in finite_issues(item, f"{prefix}.{key}")
        ]
    if isinstance(value, list):
        return [
            issue
            for index, item in enumerate(value)
            for issue in finite_issues(item, f"{prefix}[{index}]")
        ]
    return []


def add_check(checks, name, condition, details, optional=False):
    checks.append(
        {
            "name": name,
            "status": "pass" if condition else "warning" if optional else "fail",
            "details": details,
        }
    )


def heldout_comparison(model, reference):
    """Allow a quality comparison only when both complete protocols agree."""
    keys = (
        "tokenizer",
        "context_length",
        "validation_sha256",
        "precision",
        "evaluation_protocol",
        "scored_tokens",
        "evaluated_windows",
        "available_windows",
    )
    if not model or not reference:
        return None
    if not all(report.get("complete_window_sweep") for report in (model, reference)):
        return None
    if any(model.get(key) is None or model.get(key) != reference.get(key) for key in keys):
        return None
    if reference.get("checkpoint") != "openai-community/gpt2":
        return None
    if any(
        not math.isfinite(report.get(key, float("nan"))) or report[key] <= 0
        for report in (model, reference)
        for key in ("loss", "perplexity")
    ):
        return None
    return {
        "trained_model": model,
        "released_gpt2": reference,
        "loss_difference_model_minus_reference": model["loss"] - reference["loss"],
        "perplexity_ratio_model_to_reference": model["perplexity"] / reference["perplexity"],
        "interpretation": "Same held-out corpus and evaluator; lower loss/perplexity is better.",
    }


def hellaswag_comparison(model, reference, checks):
    """Retain all three declared scoring conventions and their exact counts."""
    keys = (
        "task",
        "split",
        "fewshot",
        "dataset",
        "dataset_revision",
        "validation_rows_sha256",
        "evaluated_examples",
        "available_examples",
        "tokenizer",
        "context_length",
        "precision",
        "evaluation_protocol",
    )
    if not model or not reference:
        return None
    complete = all(report.get("complete_validation_sweep") for report in (model, reference))
    matched = complete and all(
        model.get(key) is not None and model.get(key) == reference.get(key) for key in keys
    )
    matched = matched and model.get("evaluated_examples") == 10042
    matched = matched and model.get("task") == "hellaswag" and model.get("fewshot") == 0
    matched = matched and reference.get("checkpoint") == "openai-community/gpt2"
    add_check(
        checks,
        "hellaswag_comparable_complete_sweeps",
        matched,
        "Both models must score all 10,042 questions using the same dataset and protocol.",
    )
    if not matched:
        return None
    metrics = {
        "sum": "accuracy",
        "token_normalized": "accuracy_token_normalized",
        "character_normalized": "accuracy_character_normalized",
    }
    output = {
        "examples": model["evaluated_examples"],
        "chance_accuracy": 0.25,
        "interval_method": "Two-sided 95 percent binomial Wilson score interval",
        "interval_scope": "Question sampling, not training-seed variability or paired significance",
        "models": {},
    }
    for label, report in (("trained_model", model), ("released_gpt2", reference)):
        scores = {}
        for key, accuracy_key in metrics.items():
            total = report["evaluated_examples"]
            correct = report.get("correct_counts", {}).get(key)
            valid = isinstance(correct, int) and 0 <= correct <= total
            valid = valid and math.isclose(report.get(accuracy_key, -1), correct / total)
            add_check(
                checks,
                f"hellaswag_{label}_{key}_counts",
                valid,
                "Reported accuracy must equal the exact integer success count divided by questions.",
            )
            if not valid:
                return None
            scores[key] = {
                "correct": correct,
                "total": total,
                "accuracy": correct / total,
                "wilson_95": wilson_interval(correct, total),
            }
        output["models"][label] = scores
    return output


def audit_report(report):
    """Check token counts, evaluation protocols, and report file hashes."""
    manifest = read_json(report / "manifest.json")
    records = read_records(report / "metrics.jsonl")
    sessions = (
        read_records(report / "sessions.jsonl") if (report / "sessions.jsonl").is_file() else []
    )
    checks = []
    train = [row for row in records if row.get("kind") == "train"]
    validation = [row for row in records if row.get("kind") == "val"]
    checkpoints = [row for row in records if row.get("kind") == "checkpoint"]
    if not train or not validation:
        raise ValueError("Report requires measured training and validation records")
    args, config, meta = manifest["args"], manifest["config"], manifest["data_meta"]
    per_step = manifest["tokens_per_step"]
    target = args["tokens"]
    final_step = math.ceil(target / per_step)
    actual_tokens = final_step * per_step
    last_train = max(train, key=lambda row: row["step"])
    last_validation = max(validation, key=lambda row: row["step"])
    problems = finite_issues(records, "metrics")
    add_check(checks, "all_logged_metrics_finite", not problems, {"nonfinite_fields": problems})
    add_check(
        checks,
        "recorded_token_accounting",
        all(row.get("tokens_seen") == row["step"] * per_step for row in train + validation),
        "Each recorded token count must match optimizer updates times global token batch.",
    )
    add_check(
        checks,
        "final_training_budget_complete",
        last_train["step"] == final_step and last_train["tokens_seen"] == actual_tokens,
        {"expected_step": final_step, "expected_processed_tokens": actual_tokens},
    )
    add_check(
        checks,
        "final_sampled_validation_recorded",
        last_validation["step"] == final_step,
        {"last_validation_step": last_validation["step"], "expected_step": final_step},
    )
    add_check(
        checks,
        "training_records_ordered_unique",
        [row["step"] for row in train] == sorted({row["step"] for row in train}),
        "Recorded updates must form one ordered trajectory with no duplicate steps.",
    )
    completed = [row for row in sessions if row.get("event") == "end"]
    add_check(
        checks,
        "completed_training_session",
        any(row.get("complete") and row.get("step") == final_step for row in completed),
        "A completed session must independently record the expected final update.",
    )

    def optional_json(name):
        return read_json(report / name) if (report / name).is_file() else None

    evaluation = optional_json("evaluation.json")
    reference = optional_json("reference_evaluation.json")
    block = config["block_size"]
    val_tokens = meta["tokens"]["val"]
    val_windows = (val_tokens - 1) // block
    expected_scored = val_windows * block
    if evaluation:
        problems = finite_issues(evaluation, "evaluation")
        add_check(checks, "heldout_evaluation_finite", not problems, problems)
        coverage = (
            evaluation.get("complete_window_sweep")
            and evaluation.get("evaluated_windows") == val_windows
            and evaluation.get("available_windows") == val_windows
            and evaluation.get("scored_tokens") == expected_scored
            and evaluation.get("unscored_token_tail") == val_tokens - 1 - expected_scored
            and evaluation.get("validation_sha256") == meta["sha256"]["val"]
            and evaluation.get("tokenizer") == meta["tokenizer"]
            and evaluation.get("context_length") == block
        )
        add_check(
            checks,
            "exact_heldout_coverage",
            bool(coverage),
            {
                "expected_windows": val_windows,
                "expected_scored_tokens": expected_scored,
                "expected_unscored_tail": val_tokens - 1 - expected_scored,
            },
        )
        add_check(
            checks,
            "heldout_perplexity_matches_loss",
            math.isclose(evaluation["perplexity"], math.exp(evaluation["loss"]), rel_tol=1e-9),
            "Perplexity must be exp(mean negative log likelihood in nats per target token).",
        )
    else:
        add_check(checks, "complete_heldout_evaluation_present", False, "evaluation.json missing")
    heldout = heldout_comparison(evaluation, reference)
    add_check(
        checks,
        "heldout_reference_protocol_matches",
        heldout is not None,
        "Reference and trained checkpoint must use identical complete held-out protocols.",
        optional=True,
    )
    selected_validation = None
    if evaluation:
        candidates = [row for row in validation if row["step"] == evaluation.get("step")]
        selected_validation = candidates[-1] if candidates else None
        if evaluation.get("checkpoint") == "best.pt":
            best_validation = min(validation, key=lambda row: row["loss"])
            add_check(
                checks,
                "best_checkpoint_matches_validation_selection",
                evaluation.get("step") == best_validation["step"],
                {
                    "selected_step": evaluation.get("step"),
                    "minimum_sampled_validation_step": best_validation["step"],
                },
            )
    hellaswag_model = optional_json("hellaswag.json")
    hellaswag_reference = optional_json("hellaswag_reference.json")
    hellaswag = hellaswag_comparison(hellaswag_model, hellaswag_reference, checks)
    if hellaswag_model and evaluation:
        add_check(
            checks,
            "hellaswag_uses_selected_checkpoint",
            all(
                hellaswag_model.get(key) == evaluation.get(key)
                for key in ("checkpoint", "step", "training_tokens")
            ),
            "HellaSwag and the complete held-out sweep must describe the same selected checkpoint.",
        )
        inventory = optional_json("checkpoint_inventory.json")
        if inventory:
            selected = [
                row
                for row in inventory.get("files", [])
                if row.get("path") == evaluation.get("checkpoint")
            ]
            add_check(
                checks,
                "hellaswag_checkpoint_hash_matches_inventory",
                len(selected) == 1
                and hellaswag_model.get("checkpoint_sha256") == selected[0].get("sha256"),
                "The evaluator's checkpoint digest must equal the exported checkpoint inventory.",
            )
    if hellaswag is None:
        add_check(
            checks, "hellaswag_comparison_available", False, "No valid comparison", optional=True
        )
    samples = read_records(report / "samples.jsonl") if (report / "samples.jsonl").is_file() else []
    suite_valid = bool(samples) and all(
        isinstance(row.get("prompt"), str)
        and isinstance(row.get("text"), str)
        and bool(row["text"])
        and all(key in row for key in ("seed", "temperature", "top_k", "max_new_tokens"))
        and evaluation is not None
        and row.get("checkpoint") == evaluation.get("checkpoint")
        and row.get("step") == evaluation.get("step")
        for row in samples
    )
    add_check(
        checks, "sample_suite_checkpoint_and_settings", suite_valid, {"samples": len(samples)}
    )

    # Milestones use the configured token targets. A nearby sampled validation
    # measurement is deliberately not relabeled as a full-corpus evaluation.
    milestone_rows = []
    targets = sorted({0, *args.get("milestone_tokens", []), target})
    for tokens in targets:
        expected_step = math.ceil(tokens / per_step)
        matches = [row for row in validation if row["step"] == expected_step]
        if matches:
            row = matches[-1]
            milestone_rows.append(
                {
                    "target_tokens": tokens,
                    "step": row["step"],
                    "processed_tokens": row["tokens_seen"],
                    "sampled_validation_loss": row["loss"],
                    "sampled_validation_perplexity": math.exp(row["loss"]),
                }
            )
    train_windows = (meta["tokens"]["train"] - 1) // block
    rank_window_batch = args["batch_size"] * manifest["world_size"]
    first_epoch_tokens = train_windows // rank_window_batch * rank_window_batch * block
    logged_speeds = [row["tokens_per_second"] for row in train[1:]] or [
        train[0]["tokens_per_second"]
    ]
    source_names = [
        "manifest.json",
        "metrics.jsonl",
        "sessions.jsonl",
        "evaluation.json",
        "reference_evaluation.json",
        "hellaswag.json",
        "hellaswag_reference.json",
        "samples.jsonl",
        "artifacts.json",
        "provenance/executed-source.zip",
    ]
    artifacts = optional_json("artifacts.json")
    if artifacts:
        invalid = []
        for name, digest in artifacts.get("files_sha256", {}).items():
            path = report / name
            # Hash inventories are relative paths. Reject traversal before reads.
            if path.resolve().is_relative_to(report.resolve()):
                if not path.is_file() or sha256(path) != digest:
                    invalid.append(name)
            else:
                invalid.append(name)
        add_check(checks, "exported_artifact_hashes", not invalid, {"mismatched_files": invalid})
    executed_archive = report / "provenance/executed-source.zip"
    if executed_archive.is_file():
        provenance = manifest.get("provenance", {})
        passed, details = retained_source_integrity(
            executed_archive, provenance.get("source_sha256", {})
        )
        # Curated source provenance covers the retained subset of executed files.
        scope = provenance.get("source_archive_scope")
        details["scope"] = scope or {"selection": "Retained executed Python modules"}
        if scope is not None:
            original = scope.get("original_module_count") if isinstance(scope, dict) else None
            retained = scope.get("retained_module_count") if isinstance(scope, dict) else None
            omitted = (
                scope.get("omitted_operational_module_count") if isinstance(scope, dict) else None
            )
            counts_valid = (
                all(
                    isinstance(value, int) and not isinstance(value, bool) and value >= 0
                    for value in (original, retained, omitted)
                )
                and retained == details["expected_files"]
                and original == retained + omitted
            )
            details["scope_counts_match_manifest"] = counts_valid
            passed = passed and counts_valid
        add_check(
            checks,
            "retained_executed_sources_match_training",
            passed,
            details,
        )
    trainer_wall_seconds = sum(row.get("wall_seconds", 0) for row in completed)
    output = {
        "schema_version": 1,
        "audit_passed": not any(check["status"] == "fail" for check in checks),
        "checks": checks,
        "source_files_sha256": {
            name: sha256(report / name) for name in source_names if (report / name).is_file()
        },
        "measurement_scope": {
            "record_counts": dict(Counter(row.get("kind") for row in records)),
            "training_log_interval_updates": args["log_interval"],
            "finite_check_scope": "Every numerical field in every available logged record",
            "not_observed": "Unlogged per-update loss, duration, gradient norm and memory",
            "processed_tokens_expected": actual_tokens,
            "target_tokens": target,
            "expected_updates": final_step,
        },
        "data_coverage": {
            "source_documents": meta["documents"],
            "training_corpus_tokens": meta["tokens"]["train"],
            "validation_corpus_tokens": val_tokens,
            "first_epoch_training_input_tokens": first_epoch_tokens,
            "discarded_input_windows_per_epoch": train_windows % rank_window_batch,
            "processed_tokens_beyond_first_epoch": max(0, actual_tokens - first_epoch_tokens),
            "heldout_windows_expected": val_windows,
            "heldout_scored_tokens_expected": expected_scored,
            "heldout_unscored_tail_tokens": val_tokens - 1 - expected_scored,
            "sampled_validation_scored_tokens_per_measurement": (
                args["eval_batches"] * rank_window_batch * block
            ),
            "training_validation_protocol": "Fixed seeded random windows sampled with replacement",
            "final_evaluation_protocol": "Contiguous contexts, next-token targets, EOT boundaries unmasked",
            "split_method": meta["split_method"],
            "split_scope": "Exact identical document bytes share a split; no new near-duplicate audit",
        },
        "convergence": {
            "initial_sampled_validation_loss": validation[0]["loss"],
            "final_sampled_validation_loss": last_validation["loss"],
            "last_training_batch_loss": last_train["loss"],
            "full_heldout_minus_final_sampled_loss": (
                evaluation["loss"] - last_validation["loss"] if evaluation else None
            ),
            "full_heldout_minus_selected_sampled_loss": (
                evaluation["loss"] - selected_validation["loss"] if selected_validation else None
            ),
            "sampled_validation_increases": sum(
                after["loss"] > before["loss"] for before, after in zip(validation, validation[1:])
            ),
            "interpretation": "Individual training batches can fluctuate; monotonicity is not required.",
            "milestones": milestone_rows,
        },
        "runtime": {
            "completed_session_wall_seconds": trainer_wall_seconds,
            "completed_trainer_tokens_per_second": (
                actual_tokens / trainer_wall_seconds
                if trainer_wall_seconds and last_train["step"] == final_step
                else None
            ),
            "completed_trainer_timing_scope": (
                "Completed trainer sessions, including their setup, validation and checkpoints; "
                "excludes inter-session gaps, separate evaluation and other pod activity"
            ),
            "recorded_validation_seconds": sum(row["validation_seconds"] for row in validation),
            "recorded_checkpoint_seconds": sum(row["checkpoint_seconds"] for row in checkpoints),
            "full_heldout_evaluation_seconds": evaluation.get("seconds") if evaluation else None,
            "reference_heldout_evaluation_seconds": reference.get("seconds") if reference else None,
            "hellaswag_evaluation_seconds": hellaswag_model.get("seconds")
            if hellaswag_model
            else None,
            "hellaswag_reference_evaluation_seconds": (
                hellaswag_reference.get("seconds") if hellaswag_reference else None
            ),
            "logged_update_tokens_per_second": distribution(logged_speeds),
            "throughput_scope": "Logged optimizer updates, excluding validation and checkpoint time",
            "peak_allocated_cuda_bytes_rank0": max(
                (row.get("peak_cuda_memory_bytes") or 0 for row in train), default=0
            ),
            "memory_scope": "Rank 0 peak allocated tensors, not total GPU or reserved memory",
            "logged_preclipping_gradient_norm": distribution([row["grad_norm"] for row in train]),
            "logged_updates_above_clipping_threshold": sum(row["grad_norm"] > 1.0 for row in train),
            "phase_scope": "Reported phase durations only; missing update durations are not extrapolated",
        },
        "heldout_comparison": heldout,
        "hellaswag_comparison": hellaswag,
        "selected_checkpoint": {
            "filename": evaluation.get("checkpoint") if evaluation else None,
            "step": evaluation.get("step") if evaluation else None,
            "training_tokens": evaluation.get("training_tokens") if evaluation else None,
            "sampled_validation_loss": selected_validation["loss"] if selected_validation else None,
            "selection_rule": "Minimum fixed-window validation loss among saved evaluation points",
            "scope": "Evaluated checkpoint may precede the completed run's last update",
        },
        "samples": {
            "count": len(samples),
            "checkpoint": evaluation.get("checkpoint") if evaluation else None,
            "evaluation_step": evaluation.get("step") if evaluation else None,
            "scope": "Fixed-prompt qualitative examples, not a broad capability benchmark",
        },
    }
    return output


def render_figures(analysis, output):
    """Render protocol-gated comparisons with readable, restrained styling."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#d8e0e8",
            "axes.labelcolor": "#374151",
            "text.color": "#172536",
            "xtick.color": "#526272",
            "ytick.color": "#526272",
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )
    directory = output / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    stems = []

    def save(figure, stem):
        for suffix in ("svg", "png"):
            figure.savefig(directory / f"{stem}.{suffix}", dpi=200, bbox_inches="tight")
        plt.close(figure)
        stems.append(stem)

    comparison = analysis["heldout_comparison"]
    if comparison:
        reports = [comparison["trained_model"], comparison["released_gpt2"]]
        figure, axes = plt.subplots(1, 2, figsize=(10, 4.9))
        labels = ["From scratch", "Released GPT-2"]
        for axis, key, title in zip(axes, ("loss", "perplexity"), ("Cross-entropy", "Perplexity")):
            values = [report[key] for report in reports]
            bars = axis.bar(labels, values, color=["#245B9B", "#8499ad"], width=0.58)
            axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=7)
            axis.set_ylim(0, max(values) * 1.2)
            axis.set_title(title, loc="left", fontweight="bold", pad=18)
            axis.set_ylabel("Nats per target token" if key == "loss" else "exp(cross-entropy)")
            axis.grid(axis="y", alpha=0.15)
            axis.set_axisbelow(True)
        count = reports[0]["scored_tokens"]
        figure.suptitle(
            "Complete held-out comparison", x=0.06, ha="left", fontweight="bold", fontsize=16
        )
        figure.text(
            0.06,
            0.045,
            f"{count:,} target tokens  ·  Identical corpus and scoring protocol  ·  Lower is better",
            fontsize=10,
        )
        figure.tight_layout(rect=(0, 0.1, 1, 0.9))
        save(figure, "heldout_comparison")

    hellaswag = analysis["hellaswag_comparison"]
    if hellaswag:
        figure, axis = plt.subplots(figsize=(10, 5.4))
        keys = ("sum", "token_normalized", "character_normalized")
        width = 0.34
        all_high = []
        for offset, label, color, name in (
            (-width / 2, "trained_model", "#245B9B", "From scratch"),
            (width / 2, "released_gpt2", "#8499ad", "Released GPT-2"),
        ):
            scores = [hellaswag["models"][label][key] for key in keys]
            values = [row["accuracy"] for row in scores]
            low = [value - row["wilson_95"]["lower"] for value, row in zip(values, scores)]
            high = [row["wilson_95"]["upper"] - value for value, row in zip(values, scores)]
            all_high.extend(row["wilson_95"]["upper"] for row in scores)
            bars = axis.bar(
                [index + offset for index in range(3)],
                values,
                width,
                color=color,
                label=name,
                yerr=[low, high],
                capsize=4,
                error_kw={"linewidth": 1.1, "ecolor": "#374151"},
            )
            axis.bar_label(
                bars, labels=[f"{value:.1%}" for value in values], padding=13, fontsize=10
            )
        axis.axhline(0.25, color="#a76532", linestyle="--", linewidth=1.2, label="Chance 25%")
        axis.set_xticks(
            range(3), ["Raw ending likelihood", "Per ending token", "Per ending character"]
        )
        axis.set_ylim(0, max(0.42, max(all_high) * 1.3))
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.set_ylabel("Correct answers")
        axis.grid(axis="y", alpha=0.15)
        axis.set_axisbelow(True)
        axis.set_title("Zero-shot HellaSwag", loc="left", fontsize=16, fontweight="bold", pad=22)
        axis.legend(frameon=False, loc="upper right", ncols=3, fontsize=9)
        figure.text(
            0.08,
            0.025,
            "All 10,042 validation questions  ·  Wilson 95% intervals  ·  Same evaluator for both models",
            fontsize=10,
        )
        figure.tight_layout(rect=(0, 0.075, 1, 1))
        save(figure, "hellaswag_uncertainty")
    return stems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, help="Derived outputs; defaults to the report directory"
    )
    parser.add_argument("--no-figures", action="store_true", help="Perform numerical audit only")
    args = parser.parse_args()
    output = args.out or args.report
    output.mkdir(parents=True, exist_ok=True)
    analysis = audit_report(args.report)
    analysis["derived_figures"] = [] if args.no_figures else render_figures(analysis, output)
    (output / "analysis.json").write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n")
    milestones = analysis["convergence"]["milestones"]
    if milestones:
        with (output / "milestones.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(milestones[0]))
            writer.writeheader()
            writer.writerows(milestones)
    for check in analysis["checks"]:
        print(f"{check['status'].upper()}  {check['name']}")
    print(
        f"Derived {len(analysis['derived_figures'])} comparison figures and {len(milestones)} milestones"
    )
    return 0 if analysis["audit_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
