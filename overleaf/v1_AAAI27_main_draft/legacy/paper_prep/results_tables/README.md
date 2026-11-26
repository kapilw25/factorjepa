# 📊 `paper_prep/results_tables/` — per-technique head-to-head numbers

> Open items for the submission live in [`../TODO.md`](../TODO.md).

Generated **2026-07-22** by `src/utils/pairwise_table.py`. Every number here is read from
paired-bootstrap blocks that the eval pipeline already wrote; **nothing is recomputed or
re-estimated**, and no GPU work was involved.

---

## 🎯 Why this folder exists

The six `metrics_watch` figures (`forest_plot_*`, `eval_scorecard_combined`,
`scale_replication`, `scale_poc_vs_full_*`) all read one file type — `eval_metrics.json` —
and all collapse the competitor field into a single **"best competitor"** mark. That is why
the paper can only say *"against the strongest non-FactorJEPA adaptation"* and has no table
naming LoRA, DoRA, Auto-RGN, LP-FT, full-FT and continual-SSL individually.

Those per-technique comparisons were computed all along, with paired BCa bootstrap on shared
clips, and left unread inside the stage roll-up JSONs. This folder extracts them.

---

## 📁 Files

| file | rows | contents |
|---|---|---|
| `pairwise_all.csv` | 3,317 | every ordered arm pair × every metric × every scale |
| `ours_vs_technique.csv` | 1,374 | the OURS × COMPETITOR subset (the head-to-head numbers) |
| `arm_vs_frozen.csv` | 548 | every arm vs the frozen backbone (the paper's headline framing) |
| `per_technique_tables.txt` | — | rendered ASCII tables, one block per (OURS arm × scale) |
| `per_technique_tables.tex` | — | the same as `booktabs` LaTeX, ready to `\input` |

### CSV columns

| column | meaning |
|---|---|
| `scale` | `POC-2B` (ViT-G, 10k), `POC-1B` (ViT-g, 10k), `FULL-1B` (ViT-g, 116k) |
| `metric` | registry key from `configs/metric_names.json` |
| `direction` | `higher` / `lower` / `signed` (`signed` metrics have no better direction) |
| `unit` | `pp` = percentage points (Action top-1 only), `raw` = native metric units |
| `arm_a`, `arm_b` | backbone prefix stripped, so 2B and 1B key on the same token |
| `side_a`, `side_b` | `OURS` / `COMP` / `REF` (frozen) |
| `n` | shared clips behind the paired bootstrap |
| `raw_delta_a_minus_b`, `raw_ci_lo`, `raw_ci_hi` | verbatim from the source file, unoriented |
| `a_advantage`, `a_adv_ci_lo`, `a_adv_ci_hi` | **sign-oriented so positive = `arm_a` is better** |
| `separated` | `True` when the oriented 95% CI excludes zero |
| `p_value` | as stored by the source file |
| `source` | exact file (and metric sub-block) the row came from |

---

## 🔗 Source files

Per eval root, from `m12a` / `m12b` / `m12d` / `m12e` / `m12f`:

```text
probe_action/probe_paired_delta.json                    -> act
probe_motion_cos/probe_motion_cos_paired.json           -> mcos
probe_future_mse/probe_future_mse_per_variant.json      -> fut
predictor_temporal/predictor_temporal_per_variant.json  -> rollout causal tdist
                                                           teacher_free maskratio order
encoder_temporal/encoder_temporal_per_variant.json      -> aot tov pace
```

Eval roots:

```text
POC-2B   outputs/poc/vjepa_2_1_vitG_2B/eval/eval_10k     20 arms · n_test = 1,825
POC-1B   outputs/poc/vjepa_2_1_vitg_1B/eval/eval_10k     14 arms · n_test = 1,825
FULL-1B  outputs/full/vjepa_2_1_vitg_1B/eval/full         3 arms · n_test = 23,106
```

⚠️ **Deliberately excluded**, because neither yields one scalar per pair:
`probe_taxonomy/per_dim_acc.json` (15 independent per-dimension blocks) and
`encoder_temporal` → `tcc` (nested `cycle_back` / `kendalls_tau`, and only 2 pairs exported).

---

## 🧭 Sign convention

Source keys read `<armA>_minus_<armB>` and store the **raw** metric difference. Four of the
five metrics in the headline tables are *lower-is-better*, so a raw positive number means
*worse*. Every row therefore carries `a_advantage`, which flips the sign for
`direction == "lower"`:

```text
a_advantage = (A - B)      when higher is better   (act, mcos)
a_advantage = -(A - B)     when lower  is better   (fut, causal, maskratio, rollout,
                                                    tdist, teacher_free)
a_advantage = None         when direction is "signed" (order)
```

`lookup()` also flips a pair stored in the opposite order, so a table cell is always
"our arm minus that competitor".

Emphasis in the tables:

```text
ASCII   *            the oriented 95% CI excludes zero (sign shows which way)
LaTeX   \textbf{}    separated IN OUR FAVOUR
        \underline{} separated AGAINST US
        plain        CIs overlap — no separation either way
```

---

## 👥 Roster: who counts as OURS

Mirrors `src/m13_eval_plot._xb_is_ours` exactly, so these tables and the forest plots agree:

> `OURS` = `configs/arm_registry.yaml` group ∈ {`ours_flagship`, `ours_head`, `improvement`},
> **plus an explicit exception for `surgery_raw_encoder`.**

⚠️ `surgery_raw_encoder` and `surgical_autorgn_encoder` share the registry group
`surgery_ablation`, but the first is placed on the OURS side by that exception and the second
on the COMPETITOR side. That single line decides four of the eight headline forest values.

**Competitor techniques appearing in the tables:**

| arm | technique |
|---|---|
| `full_ft_encoder` | full fine-tuning, all encoder blocks trainable |
| `lpft_encoder` | LP-FT (linear probe, then fine-tune) |
| `peft_lora_encoder` | LoRA, r=16, α=32, on `qkv/proj/fc1/fc2` |
| `peft_dora_encoder` | DoRA (weight-decomposed LoRA) |
| `surgical_autorgn_encoder` | Auto-RGN surgical fine-tuning (relative-gradient-norm block selection) |
| `pretrain_encoder` | vanilla continual SSL on raw clips |

`cassle_encoder`, `ewc_encoder`, `pretrain_2X_encoder` and `pretrain_head` were trained but
never evaluated at these scales, so they produce no rows.

---

## 📌 What the tables show

**1. The prediction metrics are a clean, uniform sweep.** On `future-frame L1` and
`causal future-block L1`, FactorJEPA separates in its favour against **all six** competitor
techniques, at **both** POC scales, with no exceptions. That is 24 wins out of 24, and it is a
far stronger statement than the current "+1.7% vs the best competitor".

**2. Motion-cosine is where the competitors win at 10k**, and it is specifically full-FT and
the PEFT family — not LP-FT, and not continual-SSL at 1B. The blanket
"factorization concedes motion" framing is coarser than the data.

**3. Action top-1 is genuinely mixed** and scale-dependent: at 2B, FactorJEPA loses to
full-FT, DoRA, continual-SSL and Auto-RGN; at 1B, several of those flip to non-separated or
to wins. Any single-number claim here will be wrong somewhere.

**4. Mask-ratio robustness is not uniform** — at 2B the flagship loses to LoRA and DoRA
(−0.0031, −0.0003) while beating everything else, so the "greater robustness under reduced
visual evidence" headline holds against frozen but not against every technique.

**5. At full scale there is exactly one competitor.** `FULL-1B` contains only
frozen + LoRA + `diheavy`, so the full-scale row is literally *diheavy vs LoRA*. Every
full-scale "vs competitors" statement must be scoped to LoRA.

**6. `surgery_raw_encoder` (FactorJEPA-RAW) is competitive with, and on several cells better
than, the supervised-curriculum flagship** — visible by comparing its table block against
`surgical_3stage_DI_encoder`'s. See §3.1 of `../AUDIT_delivered_vs_claimed.md`.

---

## 🔁 Regenerate

```bash
python -u src/utils/pairwise_table.py \
    --eval-root  POC-2B=outputs/poc/vjepa_2_1_vitG_2B/eval/eval_10k \
    --eval-root  POC-1B=outputs/poc/vjepa_2_1_vitg_1B/eval/eval_10k \
    --eval-root  FULL-1B=outputs/full/vjepa_2_1_vitg_1B/eval/full \
    --metric-names configs/metric_names.json \
    --arm-registry configs/arm_registry.yaml \
    --headline act,fut,causal,mcos,maskratio \
    --ours-arm surgical_3stage_DI_encoder \
    --ours-arm surgery_raw_encoder \
    --ours-arm surgical_3stage_DI_diheavy_encoder \
    --out-dir overleaf/2026___FactorJEPA_AAAI/paper_prep/results_tables \
    2>&1 | tee logs/pairwise_table_$(date +%Y%m%d_%H%M%S).log
```

`--headline` accepts any keys from `configs/metric_names.json`; the CSVs always contain all
12 extractable metrics regardless of what the tables display. Add `--ours-arm` to emit another
table block. The extractor **raises** on an unrecognised pairwise schema rather than skipping
the entry, so a future change to a roll-up writer fails loudly here.

---

## ✅ Verification

The five `FULL-1B diheavy vs peft_lora` cells were recomputed independently from
`outputs/full/.../eval_metrics.json` means and match to 4 decimal places:

| metric | from `eval_metrics.json` | table |
|---|---|---|
| Action top-1 | −0.5064 pp | −0.5064 pp |
| future-frame L1 | +0.0164 | +0.0164 |
| causal future-block L1 | +0.0096 | +0.0096 |
| motion-cosine separation | +0.0196 | +0.0196 |
| mask-ratio robustness | +0.0103 | +0.0103 |
