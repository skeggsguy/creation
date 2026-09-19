# lm-eval results — run01 @ step_366211

_generated 2026-09-19 02:39:51Z_

## The model

| | |
|---|---|
| checkpoint | `/Users/tom/dev/llm-mod/runs/run01/ckpt/step_366211` |
| step | 366,211 |
| tokens seen | 6.00B |
| parameters | 303.48M (tied embeddings) |
| architecture | 22L / d_model 1024 / 16 heads / SwiGLU 2752 / RMSNorm / RoPE θ=10000 |
| vocab / context | 24576 / 1024 |
| HF export | `/Users/tom/dev/llm-mod/runs/hf_export/step_366211` |
| eval device | mps, float32 |

## Export verification

The MLX checkpoint and the exported HuggingFace model were fed identical tokens and their next-token logits compared. The **fp32/cpu** column isolates the weight mapping — both sides in true IEEE float32, so any real difference here is a bug. The **bf16/gpu** column is the path the model actually runs on, measured against that fp32 reference: it is the rounding we accept for free, not an error.

| probe | tokens | max abs Δlogit (fp32/cpu) | mean abs Δ (fp32/cpu) | max abs Δ (bf16/gpu) |
|---|---:|---:|---:|---:|
| `The meaning of a good life is…` | 7 | 7.915e-05 | 1.719e-05 | 6.992e-01 |
| `In the year 3000, humanity finally discovered …` | 13 | 2.995e-04 | 2.767e-05 | 2.813e+00 |
| `Cognitive behavioural therapy teaches that our…` | 12 | 1.364e-04 | 1.683e-05 | 1.577e+00 |
| corpus sample (general_val) | 200 | 4.120e-04 | 1.751e-05 | 4.105e+00 |

- next-token **argmax agreement** over 200 positions: **100.0%** (fp32/cpu), 95.0% (bf16/gpu) — gate is ≥95%
- logit spread for scale: std 30.85
- **RoPE:** our `mx.fast.rope(traditional=False)` is the half-split (GPT-NeoX / LLaMA) pairing, which is exactly what `LlamaForCausalLM` expects, so **no q/k row permutation was applied**. The exporter re-derives this from MLX at export time rather than trusting a comment.
- **Why the reference runs on CPU:** MLX's Metal float32 matmul is not IEEE float32 (~7.6e-4 relative error on a single 1024-dim product, against ~1.1e-6 for torch and for MLX's own CPU backend). Across 22 layers that compounds into ~2.4e-3 relative error, and on a model whose residual stream peaks above 1000 that is a *whole logit* of absolute difference — which reads exactly like a broken export. Pinning the reference forward pass to MLX's CPU stream closes the gap to ~1e-4.
- **result: PASS**

## Results

| task | metric | score | stderr |
|---|---|---:|---:|
| hellaswag | acc_norm | **35.8** | ±0.5 |
| arc_easy | acc_norm | **45.5** | ±1.0 |
| piqa | acc_norm | **64.0** | ±1.1 |
| lambada_openai | acc | **34.3** | ±0.7 |
| winogrande | acc | **51.0** | ±1.4 |
| sciq | acc | **75.0** | ±1.4 |

lambada_openai perplexity: **34.93**

## Against published models

| task | metric | **this model** | GPT-2 124M | GPT-2 355M | Pythia-160M | Pythia-410M |
|---|---|---:|---:|---:|---:|---:|
| hellaswag | acc_norm | **35.8** | 31.1 | 39.4 | 30.2* | 40.6* |
| arc_easy | acc_norm | **45.5** | 39.5 | 43.6 | 39.7* | 45.8* |
| piqa | acc_norm | **64.0** | 62.5 | 66.4 | 61.6* | 67.1* |
| lambada_openai | acc | **34.3** | 32.6 | 43.0 | 32.8* | 51.6* |
| winogrande | acc | **51.0** | 51.6 | 53.1 | 53.1* | 53.7* |
| sciq | acc | **75.0** | 75.2 | 77.4 | 74.1* | 81.1* |

`*` = published primary source (possibly a different harness version); unmarked = re-measured here under the same lm-eval 0.4.13 as our own column; `?` = untraced, treat as approximate.

**Metric discipline.** The anchors most often quoted for `arc_easy` and `piqa` (GPT-2 124M 43.8, GPT-2 355M 49.2, Pythia-160M 43.6, Pythia-410M 51.9) are **`acc`**, not `acc_norm`, and dropping them into an `acc_norm` column sets the bar 4-6 points too high on arc_easy. Every value above is stated in the metric named in its row — `acc_norm` for hellaswag/arc_easy/piqa, `acc` for lambada_openai/winogrande/sciq — the same metrics as our own column. Worth knowing in the other direction too: sciq `acc_norm` runs ~8-11 points *below* its `acc`, so a sciq anchor mistakenly read as acc_norm would flatter us badly.

**Provenance, and why it is not uniform.** The Pythia rows come from EleutherAI's own eval JSONs for the final checkpoint — as primary as these numbers get — but EleutherAI's README warns those were produced with a years-old harness commit and may not reproduce exactly on 0.4.x. The GPT-2 rows were re-measured here under lm-eval 0.4.13, the same version that produced our own column, which makes them the more directly comparable pair despite being measurements rather than literature. So the Pythia and GPT-2 rows are **not the same harness version**, and a one-to-two point difference between them carries no meaning.

**Two gaps worth naming.** EleutherAI never ran hellaswag for Pythia, so both Pythia hellaswag values are third-party (Mamba Table 3) — the metric is right, the harness is someone else's. And do not mix in Karpathy's widely-cited GPT-2 124M HellaSwag figure (0.2955): his `hellaswag.py` normalises by completion token count where lm-eval normalises by continuation byte length. Different metric, not comparable to the 31.1 above.

**Anchor sources**

- **Pythia-160M / Pythia-410M** (`*`) — EleutherAI's own published zero-shot eval JSONs for the final checkpoint (step143000), read directly: https://github.com/EleutherAI/pythia/blob/main/evals/pythia-v1/pythia-160m/zero-shot/160m_step143000.json and .../pythia-410m/zero-shot/410m_step143000.json
- **Pythia hellaswag** (`*`) — Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* (arXiv:2312.00752), Table 3. App. E.2.1 states the metric per task there (HellaSwag/ARC-c = acc_norm; LAMBADA/WinoGrande/PIQA/ARC-e = acc), so its hellaswag is the acc_norm we want — but note its piqa and arc_easy columns are `acc` and must not be lifted into this table.
- **GPT-2 124M and GPT-2 355M** (unmarked) — measured with lm-eval 0.4.13, `hf` backend, fp32, zero-shot, full splits, no `--limit`. Reproduce with: `lm_eval --model hf --model_args pretrained=gpt2-medium,dtype=float32 --tasks hellaswag,arc_easy,piqa,lambada_openai,winogrande,sciq --num_fewshot 0 --batch_size 32`. These are measurements, not literature values. Independent published corroboration exists for GPT-2 124M only (arc_easy 39.5, piqa 62.5, hellaswag 31.1 acc_norm): https://recsysml.substack.com/p/llm-evals-from-scratch-run-your-first, plus the harness's own reported `acc 0.2892 / acc_norm 0.3114` at https://github.com/EleutherAI/lm-evaluation-harness/issues/372

Anchors are a mix of published runs and re-measurements (see the legend and sources above), so harness version, prompt formatting and normalisation are worth a point or so in either direction. Compare shapes, not decimals.

## Run details

- lm-eval 0.4.13, `--model hf` on the export above, num_fewshot=0, limit=none, batch float32, wall 7.6 min
- task versions: {"arc_easy": 1.0, "hellaswag": 1.0, "lambada_openai": 1.0, "piqa": 1.0, "sciq": 1.0, "winogrande": 1.0}
- reproduce: `uv run --with accelerate python src/eval/run_lm_eval.py --ckpt /Users/tom/dev/llm-mod/runs/run01/ckpt/step_366211 --tasks hellaswag,arc_easy,piqa,lambada_openai,winogrande,sciq`
