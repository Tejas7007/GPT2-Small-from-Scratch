#!/usr/bin/env python3
"""Generate the README diagrams and training animation.

The summary figures read metrics from the training report. The architecture
diagram shows the model pipeline and the contents of one decoder block.
Example tokens illustrate causal attention and a distributed training update.

Example
    python scripts/render_visuals.py --report reports/final

Pillow draws the GIF using DejaVu Sans. The two font arguments accept other
font files. Each SVG includes a title and a screen reader description.
"""

import argparse
import json
import math
from html import escape
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
WIDTH = 960
BACKGROUND = "#1d2022"
SURFACE = "#25282a"
TEXT = "#eeebe5"
SECONDARY = "#babbb5"
RULE = "#515650"
CONNECTOR = "#989e97"
ACCENT = "#b9a27e"
ACTIVE_SURFACE = "#35332d"
FONT_STACK = "DejaVu Sans, Arial, sans-serif"


def svg_text(x, y, label, size=18, color=TEXT, weight=400, anchor="start"):
    """Format and escape an SVG text label."""
    return (
        f'<text x="{x}" y="{y}" font-family="{FONT_STACK}" '
        f'font-size="{size}" fill="{color}" font-weight="{weight}" '
        f'text-anchor="{anchor}">{escape(str(label))}</text>'
    )


def svg_box(x, y, width, height, fill=SURFACE, stroke=RULE):
    """Draw a rectangular model or token block."""
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1"/>'
    )


def svg_line(route, color=CONNECTOR, width=1.5, arrow=False, dashed=False):
    """Draw a connection with an optional arrowhead or dashed line."""
    marker = ' marker-end="url(#arrow)"' if arrow else ""
    dash = ' stroke-dasharray="4 4"' if dashed else ""
    return f'<path d="{route}" fill="none" stroke="{color}" stroke-width="{width}"{marker}{dash}/>'


def write_svg(output, name, height, title, description, elements):
    """Write an SVG with a title and a screen reader description."""
    header = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
        f'height="{height}" viewBox="0 0 {WIDTH} {height}" '
        'role="img" aria-labelledby="title desc">\n'
        f'<title id="title">{escape(title)}</title>\n'
        f'<desc id="desc">{escape(description)}</desc>\n'
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" '
        'refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{CONNECTOR}"/>'
        "</marker></defs>\n"
    )
    background = f'<rect width="{WIDTH}" height="{height}" fill="{BACKGROUND}"/>'
    (output / name).write_text(
        header + background + "\n" + "\n".join(elements) + "\n</svg>\n",
        encoding="utf-8",
    )


def read_measurements(report):
    """Load run metrics and the selected checkpoint's evaluation scores."""
    sources = {
        "run": "figures/run_summary.json",
        "evaluation": "evaluation.json",
        "hellaswag": "hellaswag.json",
    }
    measurements = {
        key: json.loads((report / name).read_text(encoding="utf-8"))
        for key, name in sources.items()
    }
    # Full-evaluation labels require all available windows and all 10,042
    # HellaSwag questions. Both scores must belong to the selected checkpoint.
    evaluation = measurements["evaluation"]
    hellaswag = measurements["hellaswag"]
    if (
        not evaluation["complete_window_sweep"]
        or evaluation["evaluated_windows"] != evaluation["available_windows"]
    ):
        raise ValueError("The selected checkpoint's heldout sweep is incomplete")
    if (
        not hellaswag["complete_validation_sweep"]
        or hellaswag["evaluated_examples"] != 10042
        or hellaswag["available_examples"] != 10042
    ):
        raise ValueError("The selected checkpoint's HellaSwag evaluation is incomplete")
    if (
        evaluation["checkpoint"] != hellaswag["checkpoint"]
        or evaluation["step"] != hellaswag["step"]
    ):
        raise ValueError("Heldout and HellaSwag scores must use the same selected checkpoint")
    return measurements


def render_hero(output, measurements):
    """Draw the project title, parameter count, tokens processed, and hardware."""
    run = measurements["run"]
    tokens = f"{run['training_tokens'] / 1e9:.4f}B"
    parameters = f"{run['parameters'] / 1e6:.1f}M"
    elements = [
        svg_text(36, 46, "LANGUAGE MODEL PRETRAINING", 14, SECONDARY),
        svg_text(36, 112, "GPT-2 from scratch", 48, TEXT, 700),
        svg_text(36, 154, "Trained from random weights on ten billion tokens.", 19, SECONDARY),
        svg_line("M 36 192 H 924", ACCENT, 1),
    ]
    for x, value, label, detail in [
        (36, parameters, "trainable parameters", "GPT-2 Small architecture"),
        (340, tokens, "training tokens processed", "Reused GPT-2 tokenizer"),
        (644, f"{run['world_size']} × H100", "80 GB GPUs", "Distributed bf16 training"),
    ]:
        elements.extend(
            [
                svg_text(x, 249, value, 34, TEXT, 700),
                svg_text(x, 282, label, 16, TEXT),
                svg_text(x, 308, detail, 14, SECONDARY),
            ]
        )
    for x in (316, 620):
        elements.append(svg_line(f"M {x} 217 V 310", RULE, 1))
    description = (
        f"A completed GPT-2 Small training run with {run['parameters']:,} trainable "
        f"parameters and {run['training_tokens']:,} processed training tokens. "
        f"{run['world_size']} H100 80 GB GPUs use bf16 precision. Weights start "
        "from random initialization. Text is encoded with the GPT-2 tokenizer."
    )
    write_svg(output, "hero.svg", 345, "GPT-2 from scratch", description, elements)


def render_results(output, measurements):
    """Draw summary metrics and describe their evaluation scope."""
    run = measurements["run"]
    evaluation = measurements["evaluation"]
    hellaswag = measurements["hellaswag"]
    minutes = run["completed_session_wall_seconds"] / 60
    elements = [
        svg_text(36, 54, "Measured results", 30, TEXT, 700),
        svg_text(36, 91, "Training run and selected checkpoint scores", 17, SECONDARY),
        svg_line("M 36 118 H 924", ACCENT, 1),
    ]
    values = [
        (f"{run['training_tokens'] / 1e9:.4f}B", "Training tokens", "processed"),
        (f"{evaluation['perplexity']:.2f}", "Full heldout", "perplexity"),
        (f"{hellaswag['accuracy_token_normalized'] * 100:.2f}%", "HellaSwag", "token-normalized"),
        (f"{minutes:.1f} min", "Training time", "includes validation"),
    ]
    for index, (value, label, scope) in enumerate(values):
        x = 36 + index * 228
        elements.extend(
            [
                svg_text(x, 174, value, 30, TEXT, 700),
                svg_text(x, 210, label, 16, TEXT),
                svg_text(x, 236, scope, 15, SECONDARY),
            ]
        )
        if index:
            elements.append(svg_line(f"M {x - 20} 142 V 240", RULE, 1))
    elements.extend(
        [
            svg_line("M 36 267 H 924", RULE, 1),
            svg_text(
                36,
                302,
                f"Heldout evaluation uses {evaluation['scored_tokens']:,} tokens",
                16,
                SECONDARY,
            ),
            svg_text(
                36,
                333,
                f"HellaSwag uses all {hellaswag['evaluated_examples']:,} validation examples",
                16,
                SECONDARY,
            ),
        ]
    )
    description = (
        f"Training ended at step {run['last_step']:,} after "
        f"{run['training_tokens']:,} tokens. The checkpoint "
        f"at step {evaluation['step']:,} scored full heldout perplexity "
        f"{evaluation['perplexity']:.6f} on "
        f"{evaluation['scored_tokens']:,} tokens and token-normalized HellaSwag "
        f"accuracy {hellaswag['accuracy_token_normalized'] * 100:.6f} percent on all "
        f"{hellaswag['evaluated_examples']:,} validation examples. Training took "
        f"{minutes:.6f} minutes, including startup, validation, and checkpoint saves. "
    )
    write_svg(
        output,
        "results.svg",
        365,
        "Measured training and evaluation results",
        description,
        elements,
    )


def render_architecture(output):
    """Draw the model pipeline and the contents of one decoder block."""
    elements = [
        svg_text(36, 55, "The model", 30, TEXT, 700),
        svg_text(
            36, 91, "GPT-2 Small with learned positions and tied vocabulary weights", 17, SECONDARY
        ),
        svg_line("M 36 118 H 924", ACCENT, 1),
        svg_text(
            36,
            149,
            "12 layers   ·   12 heads   ·   768 width   ·   1024 context   ·   50,257 vocabulary",
            16,
            SECONDARY,
        ),
    ]
    x, width, center = 64, 280, 204
    overview = [
        (181, 72, "Token IDs", "[B, T]"),
        (293, 112, "Learned embeddings", "Token + position vectors"),
        (449, 78, "Decoder block × 12", "[B, T, 768]"),
        (571, 64, "Final LayerNorm", ""),
        (679, 78, "Tied output projection", "768 → 50,257"),
        (801, 78, "Next-token logits", "[B, T, 50,257]"),
    ]
    for index, (y, height, label, shape) in enumerate(overview):
        elements.extend(
            [
                svg_box(x, y, width, height),
                svg_text(center, y + 30, label, 18, TEXT, 500, "middle"),
            ]
        )
        if shape:
            elements.append(svg_text(center, y + 56, shape, 15, SECONDARY, 400, "middle"))
        if index == 1:
            elements.extend(
                [
                    svg_text(
                        center, y + 80, "Both tables are trainable", 13, SECONDARY, 400, "middle"
                    ),
                    svg_text(
                        center, y + 99, "Positions broadcast across B", 13, SECONDARY, 400, "middle"
                    ),
                ]
            )
        if index < len(overview) - 1:
            elements.append(
                svg_line(f"M {center} {y + height} V {overview[index + 1][0] - 12}", arrow=True)
            )
    elements.extend(
        [
            svg_line("M 64 331 H 43 V 718 H 64", RULE, 1.5, dashed=True),
            f'<text x="27" y="564" transform="rotate(-90 27 564)" font-family="{FONT_STACK}" font-size="13" fill="{SECONDARY}">Shared token weights</text>',
            svg_line("M 344 488 H 393", arrow=True),
            svg_box(406, 181, 518, 698, BACKGROUND),
            svg_text(428, 217, "One decoder block", 22, TEXT, 500),
            svg_text(902, 217, "× 12", 18, SECONDARY, 400, "end"),
            svg_text(666, 256, "Hidden states  [B, T, 768]", 15, SECONDARY, 400, "middle"),
            svg_line("M 666 266 V 276", arrow=True),
            svg_box(540, 284, 252, 52),
            svg_text(666, 317, "LayerNorm", 18, TEXT, 400, "middle"),
            svg_line("M 666 336 V 362", arrow=True),
            svg_box(498, 374, 336, 96),
            svg_text(666, 409, "Causal self-attention", 18, TEXT, 500, "middle"),
            svg_text(666, 441, "Q K V each [B, 12, T, 64]", 14, SECONDARY, 400, "middle"),
            svg_line("M 666 470 V 492", arrow=True),
            svg_line("M 666 271 H 450 V 510 H 648", ACCENT, 1.5, arrow=True),
            f'<text x="439" y="432" transform="rotate(-90 439 432)" font-family="{FONT_STACK}" font-size="13" fill="{SECONDARY}">Identity path</text>',
            f'<circle cx="666" cy="510" r="18" fill="{BACKGROUND}" stroke="{ACCENT}" stroke-width="1.5"/>',
            svg_text(666, 517, "+", 24, TEXT, 400, "middle"),
            svg_line("M 666 528 V 542", arrow=True),
            svg_box(540, 554, 252, 52),
            svg_text(666, 587, "LayerNorm", 18, TEXT, 400, "middle"),
            svg_line("M 666 606 V 634", arrow=True),
            svg_box(498, 646, 336, 96),
            svg_text(666, 681, "Position-wise MLP", 18, TEXT, 500, "middle"),
            svg_text(666, 713, "768 → 3072 → GELU → 768", 14, SECONDARY, 400, "middle"),
            svg_line("M 666 742 V 765", arrow=True),
            svg_line("M 666 539 H 879 V 784 H 684", ACCENT, 1.5, arrow=True),
            f'<text x="897" y="698" transform="rotate(-90 897 698)" font-family="{FONT_STACK}" font-size="13" fill="{SECONDARY}">Identity path</text>',
            f'<circle cx="666" cy="784" r="18" fill="{BACKGROUND}" stroke="{ACCENT}" stroke-width="1.5"/>',
            svg_text(666, 791, "+", 24, TEXT, 400, "middle"),
            svg_line("M 666 802 V 824", arrow=True),
            svg_text(666, 852, "Hidden states  [B, T, 768]", 15, SECONDARY, 400, "middle"),
            svg_line("M 36 909 H 924", RULE, 1),
            svg_text(36, 945, "Pre-LN decoder", 16, TEXT),
            svg_text(924, 945, "124,439,808 parameters", 16, TEXT, 400, "end"),
            svg_text(36, 978, "B is batch size   ·   T is sequence length", 15, SECONDARY),
        ]
    )
    description = (
        "Token IDs index learned token embeddings. Learned position embeddings "
        "are added and broadcast across the batch. Twelve pre-LayerNorm decoder "
        "blocks each apply causal self-attention with 12 heads of width 64 and a "
        "residual addition, then a 768 to 3072 GELU MLP and a second residual "
        "addition. A final LayerNorm and vocabulary projection tied to token "
        "embedding weights produce next-token logits. B is batch size and T is "
        "sequence length. Vocabulary size is 50,257 and context length is 1024. "
        "The model has 124,439,808 trainable parameters."
    )
    write_svg(
        output, "architecture.svg", 1010, "GPT-2 Small model architecture", description, elements
    )


def raster_text(draw, position, label, size, color, fonts, bold=False, anchor=None):
    """Place a text label at its specified size and alignment."""
    font = ImageFont.truetype(fonts[bool(bold)], size)
    draw.text(position, label, font=font, fill=color, anchor=anchor)


def raster_line(draw, points, color=CONNECTOR, width=1, arrow=False):
    """Draw a line and an optional arrowhead."""
    draw.line(points, fill=color, width=width)
    if arrow:
        start, end = points[-2:]
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        depth = 7
        draw.polygon(
            [
                end,
                (end[0] - depth * math.cos(angle - 0.48), end[1] - depth * math.sin(angle - 0.48)),
                (end[0] - depth * math.cos(angle + 0.48), end[1] - depth * math.sin(angle + 0.48)),
            ],
            fill=color,
        )


def training_labels():
    """Return the example tokens and labels used by both training diagrams."""
    return (
        ["the", "model", "learns", "from", "text"],
        ["model", "learns", "from", "text", "."],
        ["Causal forward", "Token loss", "Gradient all-reduce", "AdamW update"],
        ["Visible context only", "Shifted targets", "Four GPU average", "Synchronized replicas"],
    )


def render_animation(output, fonts):
    """Animate causal visibility, token loss, gradient averaging, and the update."""
    words, targets, stages, details = training_labels()
    positions = [210 + 139 * index for index in range(5)]
    height = 575
    frames = []
    for state in range(8):
        image = Image.new("RGB", (WIDTH, height), BACKGROUND)
        draw = ImageDraw.Draw(image)

        def put(position, label, size=18, color=TEXT, bold=False, anchor=None):
            """Draw a label in the current frame."""
            raster_text(draw, position, label, size, color, fonts, bold, anchor)

        put((36, 30), "One training update", 30, TEXT, True)
        put((36, 81), "Example tokens illustrate next-token training", 17, SECONDARY)
        raster_line(draw, [(36, 119), (924, 119)], ACCENT)
        put((36, 160), "Input tokens", 16, SECONDARY)
        put((36, 262), "Target tokens", 16, SECONDARY)
        for index, (x, word, target) in enumerate(zip(positions, words, targets)):
            visible = state < 5 and index <= state
            selected = state < 5 and index == state
            target_active = selected or state >= 5
            draw.rectangle(
                (x, 145, x + 118, 190),
                fill=ACTIVE_SURFACE if visible else BACKGROUND,
                outline=ACCENT if visible else RULE,
                width=1,
            )
            put((x + 59, 167), word, 18, TEXT if visible else SECONDARY, anchor="mm")
            raster_line(
                draw, [(x + 59, 204), (x + 59, 232)], ACCENT if target_active else RULE, arrow=True
            )
            draw.rectangle(
                (x, 247, x + 118, 292),
                fill=ACTIVE_SURFACE if target_active else BACKGROUND,
                outline=ACCENT if target_active else RULE,
                width=1,
            )
            put((x + 59, 270), target, 18, TEXT if target_active else SECONDARY, anchor="mm")
        if state == 0:
            caption = "The first input token predicts the next token"
        elif state < 5:
            caption = f"Position {state + 1} predicts its next token from input tokens 1 through {state + 1}"
        elif state == 5:
            caption = "Cross entropy scores each next-token prediction against its target"
        elif state == 6:
            caption = "Average gradients from four GPUs processing different batches"
        else:
            caption = "Apply the same optimizer update on all four GPUs"
        put((36, 330), caption, 16, SECONDARY)
        raster_line(draw, [(36, 374), (924, 374)], RULE)
        active = 0 if state < 5 else state - 4
        for index, (stage, detail) in enumerate(zip(stages, details)):
            x = 36 + 228 * index
            color = ACCENT if index == active else SECONDARY
            put((x, 397), f"{index + 1:02}", 14, color)
            put((x, 427), stage, 16, TEXT, True)
            raster_line(
                draw,
                [(x, 457), (x + 186, 457)],
                ACCENT if index == active else RULE,
                2 if index == active else 1,
            )
            put((x, 474), detail, 14, SECONDARY)
        raster_line(draw, [(36, 519), (924, 519)], RULE)
        put((36, 540), "Training predictions are parallel", 14, SECONDARY)
        put((924, 540), "Checkpoints are periodic", 14, SECONDARY, anchor="ra")
        frames.append(image)
    # One palette for every frame prevents GIF colors from shifting.
    palette = frames[0].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    encoded = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    encoded[0].save(
        output / "training-flow.gif",
        save_all=True,
        append_images=encoded[1:],
        duration=[1000] * 5 + [1250, 1250, 1850],
        loop=0,
        optimize=True,
        disposal=2,
    )


def render_static_flow(output):
    """Draw the training update as a single SVG."""
    words, targets, stages, details = training_labels()
    elements = [
        svg_text(36, 57, "One training update", 30, TEXT, 700),
        svg_text(36, 100, "Example tokens illustrate next-token training", 17, SECONDARY),
        svg_line("M 36 119 H 924", ACCENT, 1),
        svg_text(36, 179, "Input tokens", 16, SECONDARY),
        svg_text(36, 281, "Target tokens", 16, SECONDARY),
    ]
    for index, (word, target) in enumerate(zip(words, targets)):
        x = 210 + 139 * index
        elements.extend(
            [
                svg_box(x, 145, 118, 45, BACKGROUND),
                svg_text(x + 59, 174, word, 18, TEXT, 400, "middle"),
                svg_line(f"M {x + 59} 204 V 232", arrow=True),
                svg_box(x, 247, 118, 45, BACKGROUND),
                svg_text(x + 59, 277, target, 18, TEXT, 400, "middle"),
            ]
        )
    elements.extend(
        [
            svg_text(
                36,
                349,
                "Each position predicts its next token using only its visible input context",
                16,
                SECONDARY,
            ),
            svg_line("M 36 374 H 924", RULE, 1),
        ]
    )
    for index, (stage, detail) in enumerate(zip(stages, details)):
        x = 36 + 228 * index
        elements.extend(
            [
                svg_text(x, 411, f"{index + 1:02}", 14, SECONDARY),
                svg_text(x, 445, stage, 16, TEXT, 600),
                svg_line(f"M {x} 457 H {x + 186}", RULE, 1),
                svg_text(x, 489, detail, 14, SECONDARY),
            ]
        )
    elements.extend(
        [
            svg_line("M 36 519 H 924", RULE, 1),
            svg_text(36, 555, "Training predictions are parallel", 14, SECONDARY),
            svg_text(924, 555, "Checkpoints are periodic", 14, SECONDARY, 400, "end"),
        ]
    )
    description = (
        "Example tokens show input and target sequences shifted by one "
        "position. Each position uses the current and earlier input tokens "
        "to predict the next token. Cross entropy scores these predictions "
        "against their targets. Four GPUs compute "
        "gradients on distinct batches, average them by all-reduce, and apply "
        "the same AdamW update. Training predictions are computed in parallel "
        "and checkpoints are saved periodically."
    )
    write_svg(
        output,
        "training-flow.svg",
        575,
        "One distributed next-token training update",
        description,
        elements,
    )


def resolve_project_path(value):
    """Resolve a relative path from the repository root."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main():
    """Generate the five README visuals."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="reports/final", help="Exported measurement directory")
    parser.add_argument("--output", default="docs/figures", help="Destination for README assets")
    parser.add_argument(
        "--regular-font", default="DejaVuSans.ttf", help="Regular font discoverable by Pillow"
    )
    parser.add_argument(
        "--bold-font", default="DejaVuSans-Bold.ttf", help="Bold font discoverable by Pillow"
    )
    args = parser.parse_args()
    output = resolve_project_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    measurements = read_measurements(resolve_project_path(args.report))
    fonts = (args.regular_font, args.bold_font)
    render_hero(output, measurements)
    render_results(output, measurements)
    render_architecture(output)
    render_animation(output, fonts)
    render_static_flow(output)
    print("Rendered five README assets")


if __name__ == "__main__":
    main()
