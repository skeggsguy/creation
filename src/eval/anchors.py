"""Published zero-shot scores for comparably-sized open models.

A number on its own ("hellaswag 33.4") means nothing; the only useful question
about a 303M from-scratch pretrain is where it lands against models people have
actually read papers about.  These are the anchors.

Metric per task matches what the harness reports and what we score ourselves:
`acc_norm` for the multiple-choice tasks whose options differ in length
(hellaswag, arc_easy, piqa), plain `acc` for lambada_openai, winogrande and
sciq.  See `TASK_METRIC` — it is the single source of truth, shared with the
report renderer, so our column and the anchor columns cannot drift apart.

THE acc / acc_norm TRAP
=======================
This is the thing that makes most small-model comparison tables wrong, and it
bit this file once already.  The widely-circulated anchor numbers for arc_easy
and piqa are **`acc`**, not `acc_norm`, and people copy them into columns
labelled `acc_norm` without checking.  The gap is not cosmetic:

    arc_easy   GPT-2 124M    acc 43.81  vs  acc_norm 39.48
               Pythia-160M   acc 43.52  vs  acc_norm 39.65
               Pythia-410M   acc 52.10  vs  acc_norm 45.79
    sciq       GPT-2 124M    acc 75.20  vs  acc_norm 64.40

So an arc_easy anchor of "43.8" in an acc_norm column silently sets the bar 4-6
points too high, and a sciq anchor taken as acc_norm would set it ~11 points too
low.  Every value below is stated in the metric named in `TASK_METRIC` for that
task, taken from a source where the metric was explicitly identified.

HONESTY POLICY
==============
Anchor tables are where eval reports quietly turn into fiction: a number gets
copied from a blog post that copied it from a tweet, the metric changes
somewhere along the chain, and the comparison ends up measuring nothing.  So
every cell carries a confidence marker — `""` for values read out of a primary
source, `"?"` for values we could not trace and are carrying forward as
approximate.  A `?` in the report means *do not draw a conclusion from this
cell*.

Anchors are also not re-runs.  Harness version, prompt formatting and
normalisation move these by a point or so, so the honest use is comparing shapes
("clearly above the 160M class on sciq, clearly below 410M on lambada"), never
decimals.
"""

from __future__ import annotations

# task -> metric key as lm-eval reports it
TASK_METRIC = {
    "hellaswag": "acc_norm",
    "arc_easy": "acc_norm",
    "piqa": "acc_norm",
    "lambada_openai": "acc",
    "winogrande": "acc",
    "sciq": "acc",
}

# Confidence markers.  These say *where a number came from*, which at this scale
# matters as much as the number: a value measured under our own harness version is
# more comparable to our column than a published value from a harness commit of a
# different era, even though the published one is the more "official" artifact.
MARK_MEASURED = ""    # re-measured here under lm-eval 0.4.13 — same harness as our column
MARK_PUBLISHED = "*"  # published primary source, possibly a different harness version
MARK_APPROX = "?"     # untraced; approximate, do not conclude from it

LEGEND = (
    "`*` = published primary source (possibly a different harness version); "
    "unmarked = re-measured here under the same lm-eval 0.4.13 as our own column; "
    "`?` = untraced, treat as approximate."
)

# model -> {task: (percent, confidence)}
ANCHORS: dict[str, dict[str, tuple[float | None, str]]] = {
    "GPT-2 124M": {
        "hellaswag": (31.1, MARK_MEASURED),
        "arc_easy": (39.5, MARK_MEASURED),   # acc_norm 39.48; the familiar 43.8 is acc
        "piqa": (62.5, MARK_MEASURED),       # acc_norm 62.51; acc is 62.89
        "lambada_openai": (32.6, MARK_MEASURED),
        "winogrande": (51.6, MARK_MEASURED),
        "sciq": (75.2, MARK_MEASURED),       # acc; acc_norm would be 64.40
    },
    "GPT-2 355M": {
        "hellaswag": (39.4, MARK_MEASURED),  # acc_norm 0.3938; acc is 33.31
        "arc_easy": (43.6, MARK_MEASURED),   # acc_norm 0.4360; acc is 49.07
        "piqa": (66.4, MARK_MEASURED),       # acc_norm 0.6638; acc is 67.63
        "lambada_openai": (43.0, MARK_MEASURED),   # 0.4298, ppl 18.26
        "winogrande": (53.1, MARK_MEASURED),       # 0.5312
        "sciq": (77.4, MARK_MEASURED),       # acc 0.7740; acc_norm would be 67.20
    },
    "Pythia-160M": {
        "hellaswag": (30.2, MARK_PUBLISHED),   # Mamba Table 3; EleutherAI never ran it
        "arc_easy": (39.7, MARK_PUBLISHED),    # acc_norm 39.65; acc is 43.52
        "piqa": (61.6, MARK_PUBLISHED),        # acc_norm 61.64; acc is 62.73
        "lambada_openai": (32.8, MARK_PUBLISHED),
        "winogrande": (53.1, MARK_PUBLISHED),  # 53.12
        "sciq": (74.1, MARK_PUBLISHED),        # acc; acc_norm would be 66.80
    },
    "Pythia-410M": {
        "hellaswag": (40.6, MARK_PUBLISHED),   # Mamba Table 3
        "arc_easy": (45.8, MARK_PUBLISHED),    # acc_norm 45.79; acc is 52.10
        "piqa": (67.1, MARK_PUBLISHED),        # acc_norm 67.14; acc is 66.76
        "lambada_openai": (51.6, MARK_PUBLISHED),  # 51.62
        "winogrande": (53.7, MARK_PUBLISHED),      # 53.67
        "sciq": (81.1, MARK_PUBLISHED),        # acc; acc_norm would be 72.10
    },
}

SOURCES: list[str] = [
    "**Pythia-160M / Pythia-410M** (`*`) — EleutherAI's own published zero-shot eval "
    "JSONs for the final checkpoint (step143000), read directly: "
    "https://github.com/EleutherAI/pythia/blob/main/evals/pythia-v1/pythia-160m/zero-shot/160m_step143000.json "
    "and .../pythia-410m/zero-shot/410m_step143000.json",
    "**Pythia hellaswag** (`*`) — Gu & Dao, *Mamba: Linear-Time Sequence Modeling with "
    "Selective State Spaces* (arXiv:2312.00752), Table 3. App. E.2.1 states the metric "
    "per task there (HellaSwag/ARC-c = acc_norm; LAMBADA/WinoGrande/PIQA/ARC-e = acc), "
    "so its hellaswag is the acc_norm we want — but note its piqa and arc_easy columns "
    "are `acc` and must not be lifted into this table.",
    "**GPT-2 124M and GPT-2 355M** (unmarked) — measured with lm-eval 0.4.13, `hf` "
    "backend, fp32, zero-shot, full splits, no `--limit`. Reproduce with: "
    "`lm_eval --model hf --model_args pretrained=gpt2-medium,dtype=float32 --tasks "
    "hellaswag,arc_easy,piqa,lambada_openai,winogrande,sciq --num_fewshot 0 "
    "--batch_size 32`. These are measurements, not literature values. Independent "
    "published corroboration exists for GPT-2 124M only (arc_easy 39.5, piqa 62.5, "
    "hellaswag 31.1 acc_norm): "
    "https://recsysml.substack.com/p/llm-evals-from-scratch-run-your-first, plus the "
    "harness's own reported `acc 0.2892 / acc_norm 0.3114` at "
    "https://github.com/EleutherAI/lm-evaluation-harness/issues/372",
]

NOTES = (
    "**Metric discipline.** The anchors most often quoted for `arc_easy` and `piqa` "
    "(GPT-2 124M 43.8, GPT-2 355M 49.2, Pythia-160M 43.6, Pythia-410M 51.9) are "
    "**`acc`**, not `acc_norm`, and dropping them into an `acc_norm` column sets the "
    "bar 4-6 points too high on arc_easy. Every value above is stated in the metric "
    "named in its row — `acc_norm` for hellaswag/arc_easy/piqa, `acc` for "
    "lambada_openai/winogrande/sciq — the same metrics as our own column. Worth knowing "
    "in the other direction too: sciq `acc_norm` runs ~8-11 points *below* its `acc`, "
    "so a sciq anchor mistakenly read as acc_norm would flatter us badly.\n\n"
    "**Provenance, and why it is not uniform.** The Pythia rows come from EleutherAI's "
    "own eval JSONs for the final checkpoint — as primary as these numbers get — but "
    "EleutherAI's README warns those were produced with a years-old harness commit and "
    "may not reproduce exactly on 0.4.x. The GPT-2 rows were re-measured here under "
    "lm-eval 0.4.13, the same version that produced our own column, which makes them "
    "the more directly comparable pair despite being measurements rather than "
    "literature. So the Pythia and GPT-2 rows are **not the same harness version**, and "
    "a one-to-two point difference between them carries no meaning.\n\n"
    "**Two gaps worth naming.** EleutherAI never ran hellaswag for Pythia, so both "
    "Pythia hellaswag values are third-party (Mamba Table 3) — the metric is right, the "
    "harness is someone else's. And do not mix in Karpathy's widely-cited GPT-2 124M "
    "HellaSwag figure (0.2955): his `hellaswag.py` normalises by completion token "
    "count where lm-eval normalises by continuation byte length. Different metric, not "
    "comparable to the 31.1 above."
)
