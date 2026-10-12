# Generating the visual assets

[render_visuals.py](../scripts/render_visuals.py) draws the README's overview,
results panel, architecture diagram, and training animation. The results panel
reads the saved final report. The architecture follows the model's computation,
and the animation uses illustrative tokens to show a training update.

## Regenerate the assets

Install the project dependencies and run the renderer from the repository.
Relative `--report` and `--output` paths resolve from the repository root.

```bash
python -m pip install -r requirements.txt
python scripts/render_visuals.py --report reports/final
```

The default destination is `docs/figures`. Choose a different output directory
to inspect a variation alongside the current assets.

```bash
python scripts/render_visuals.py \
  --report reports/final \
  --output runs/visual-preview
```

| Asset | What it shows |
| --- | --- |
| `hero.svg` | The model dimensions and completed training scope |
| `results.svg` | Processed token count, selected-checkpoint scores, and training time |
| `architecture.svg` | The model path, an expanded pre-normalized block, residual connections, and tied vocabulary weights |
| `training-flow.svg` | Shifted targets and the distributed optimizer update in a still diagram |
| `training-flow.gif` | Causal visibility, target alignment, gradient averaging, and the synchronized update in motion |

The main README shows the still training diagram, with the animation available
in an expandable section. Both explain the same update, so a reader can follow
the computation at their own pace. SVG titles and descriptions supply screen
reader labels for the diagrams.

## Fonts and rendering

Python's standard library produces the SVG files. Pillow renders the GIF and
is included in the project requirements with the supported range `>=10,<13`.
The default fonts are `DejaVuSans.ttf` and `DejaVuSans-Bold.ttf`. Font arguments
follow Pillow's lookup rules, with relative file paths resolved from the
invocation directory.

```bash
python scripts/render_visuals.py \
  --report reports/final \
  --regular-font DejaVuSans.ttf \
  --bold-font DejaVuSans-Bold.ttf
```

The SVG font stack uses the same sans-serif typography. Graphite surfaces,
warm silver text, fine rules, and a muted bronze accent organize the page and
make tensor connections easier to follow. In the animation, motion marks the
current step of the computation. Matching the font files and Pillow version
helps reproduce the same rendering.

## Figures from the experiment

The [final report](../reports/final/figures) contains loss, runtime, optimization,
gradient diagnostics, and quality comparisons. These figures are generated
from saved measurements by [the metric plotter](../gpt2/plot_metrics.py) and
[the analysis script](../scripts/analyze_results.py). The documentation renderer
reads the report to summarize it and draws the architecture and training
process from their definitions.
