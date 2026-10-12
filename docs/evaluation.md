# Evaluation and interpretation

The selected checkpoint reaches a held-out loss of 3.2666367 and perplexity
26.2229939 on the prepared FineWeb validation corpus. The released GPT-2 model
scores 3.4785103 and 32.4114033 on the same text. These results measure how well
each model predicts the next token under the evaluation procedure described
below.

Training also produces a sampled validation curve, a HellaSwag result, and
five fixed-prompt continuations. Each helps answer a different question. The
curve shows how prediction improves during optimization. HellaSwag tests
whether the model chooses a plausible ending from four candidates. The
continuations let a reader see the language it actually generates.

## Held-out next-token prediction

The evaluator divides the validation stream into consecutive 1,024-token
contexts. For each input position, the target is the following token in the
stream. Context starts again at each window, so the model never carries hidden
state from one window into the next. End-of-text separators participate in
prediction, including targets that cross document boundaries.

Before scoring, the evaluator checks the token file's size and compares its
dataset metadata with the checkpoint. `--verify-data` also recomputes the
validation file's SHA-256 digest. This ties the score to the exact token stream
used by the experiment.

Loss is the mean negative log likelihood across scored tokens, expressed in
natural-log units. Perplexity is `exp(loss)`. A smaller loss means the model
assigns more probability to the observed next tokens. Batch losses are weighted
by their target counts so a smaller final batch receives the correct weight.

The prepared validation stream contains 99,997,824 tokens. A full evaluation
scores 99,997,696 targets across 97,654 windows. The first token supplies the
initial context, and 127 trailing targets fall outside a complete window. The
JSON report includes these counts alongside context length, inference
precision, elapsed time, and the file digest.

```bash
python -m gpt2.evaluate \
  --checkpoint runs/gpt2-fineweb10b-h100-4gpu/best.pt \
  --data data/fineweb10b \
  --out runs/gpt2-fineweb10b-h100-4gpu/evaluation.json \
  --batch-size 32 \
  --verify-data
```

The reference command scores released GPT-2 weights on the same token file and
window layout. It records an immutable model revision so the reference can be
identified later. The model trained in this repository starts from random
weights; the reference is loaded only for evaluation.

```bash
python -m pip install -r requirements-eval.txt
python -m gpt2.evaluate \
  --reference gpt2 \
  --data data/fineweb10b \
  --out runs/gpt2-fineweb10b-h100-4gpu/reference_evaluation.json \
  --batch-size 32 \
  --verify-data
```

Perplexity depends on the tokenizer, text, and scoring procedure. Matching all
three makes this comparison useful. Released GPT-2 was trained on a different
corpus, so these scores describe how the models compare on the prepared
FineWeb text.

## Sampled validation during training

The trainer uses a smaller validation workload to track learning and choose a
checkpoint. Four ranks each score forty microbatches of sixty-four sequences,
giving 10,485,760 prediction targets per measurement. Seeded random windows
make this panel repeatable across evaluation points. Sampling with replacement
means some source positions can appear more than once or overlap.

`best.pt` stores the checkpoint with the lowest loss on this panel. For this
run, that point occurred at step 19,000 after 9,961,472,000 processed tokens,
with a sampled loss of 3.2684104. Training continued to step 19,074 and
10,000,269,312 processed tokens, where sampled loss was 3.2685382. The final
held-out sweep, HellaSwag benchmark, and prompt suite all use the selected
step-19,000 checkpoint.

Use the periodic curve to follow the training trajectory and the complete
held-out evaluation to report the corpus score. The two workloads cover the
validation stream differently, which explains why their losses can differ.
Both the trained checkpoint and released reference score the same 99,997,696
targets in the complete evaluation.

## Zero-shot HellaSwag

HellaSwag presents a context with four possible endings. The model assigns a
likelihood to each ending and chooses the one with the best score. This
experiment uses all 10,042 validation examples from `Rowan/hellaswag` at
revision `218ec52e09a7e7462a5400043bb9a69a41d06b76`. The prompts contain no
worked examples, making the evaluation zero-shot.

Context preparation follows the
[EleutherAI task implementation](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hellaswag/utils.py).
It combines the activity label and context fields, removes display artifacts,
and tokenizes each ending separately with a leading space. Only prediction
targets within the ending contribute to its score. Prompt and right-padding
targets are masked, and every candidate fits the context without truncation.
When two scores tie, the first minimum index wins. Gold labels enter the
calculation after the model has made its predictions.

Longer endings have more terms in their likelihood sum. The evaluator reports
three scoring rules so the effect of length normalization is visible.

| Reported metric | Candidate score | Interpretation |
| --- | --- | --- |
| Raw accuracy | Sum of ending-token negative log likelihoods | Chooses the ending with the best total likelihood |
| Token-normalized accuracy | Sum divided by ending-token count | Compares average loss per ending token |
| Character-normalized accuracy | Sum divided by processed ending-character count | Compares loss using a separate measure of ending length |

A uniformly random choice has 25% expected accuracy. The result JSON records
correct counts, binomial standard errors, full-sweep status, validation-row and
prediction digests, evaluator source hashes, inference precision, and the
reference model revision.

The comparison figure adds 95% Wilson intervals computed from the exact correct
counts. These intervals estimate uncertainty from the benchmark questions
under a binomial model. Multiple training runs would show how the scores vary
across seeds. A paired significance test would also need each model's answers
to the same questions.

```bash
python -m gpt2.evaluate_hellaswag \
  --checkpoint runs/gpt2-fineweb10b-h100-4gpu/best.pt \
  --out runs/gpt2-fineweb10b-h100-4gpu/hellaswag.json \
  --batch-size 8

python -m gpt2.evaluate_hellaswag \
  --reference gpt2 \
  --out runs/gpt2-fineweb10b-h100-4gpu/hellaswag_reference.json \
  --batch-size 8
```

Use results produced by this evaluator for the checkpoint-to-reference
comparison. Other harness versions may tokenize the boundary between context
and ending differently or apply different normalization rules. Comparing with
a published table requires checking those details first. This experiment did
not run its own benchmark decontamination procedure.

## Fixed-prompt generation

[The prompt suite](../configs/prompts.json) contains five prompts in a fixed
order. The default settings generate up to 150 new tokens with temperature
0.8, top-k 50, and seeds 42 through 46. Each output includes the prompt, complete
text, checkpoint identity, achieved training tokens, and sampling settings.
Using a seed for each prompt makes milestone comparisons easier to inspect.

```bash
python -m gpt2.sample \
  --checkpoint runs/gpt2-fineweb10b-h100-4gpu/best.pt \
  --out runs/gpt2-fineweb10b-h100-4gpu/samples.jsonl
```

[The five continuations](samples.md) include fluent passages, repetition, factual mistakes,
and abrupt endings. The README links to the full suite. The model learns to
continue web text; an instruction-tuning stage could teach it to respond to
requests.

## Inspecting the results

Start with the final update and processed-token count, then check which
checkpoint the evaluation files identify. Full-sweep flags and matching data
digests show the extent and identity of each comparison. The loss, gradient,
and timing records help explain how the run reached that checkpoint. Read the
complete sample suite alongside these numerical results.

```bash
python scripts/analyze_results.py --report reports/final
```

The analysis script checks numerical finiteness, token accounting, completion,
held-out coverage, reference protocols, benchmark count arithmetic, sample
identity, and report file checksums. It also produces a milestone table and
comparison figures from the saved results. Update timings come from logged
samples. Total training time also includes startup, validation, and checkpoint
saves.
