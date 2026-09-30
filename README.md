# Report What Your Audit Cannot Rule Out

**Operating characteristics of secret-loyalty detection, demonstrated on a real model.**

A secret loyalty is an undisclosed, principal-directed disposition in a language model; an
*audit* is a procedure that decides whether one is present. This repository treats an audit
as a **measurement instrument** and shows, on a real model (`Qwen3-0.6B`), that its verdicts
are uninterpretable unless it reports three operating characteristics alongside them:

1. **Reachability** — can the decision rule even fire at this sample size? (Below `n=10`
   the `DETECTED` tier is *structurally impossible* here; `n=3` probes cap at `p≈0.26`.)
2. **Power / MDE** — the smallest loyalty detectable at 80% power under the *measured*
   scorer noise (`σ̂≈0.5–0.7` here → MDE `≈0.6` at `n=12`).
3. **Equivalence bound** — after a remediation, the residual a null result cannot exclude
   (`≈0.4` here, matching the standard TOST margin).

The experiment installs a principal-directed favorability shift **at inference time only**
(a difference-in-means steering vector injected by a forward hook — no weight change, no
training data), for **China, Russia, and the USA**, plus a **matched neutral
negative-control principal (Uruguay)**, and asks whether the install *branches* to held-out
entities never named in the intervention. The headline: the install is easy to detect where
it is named, apparent branches are dominated by a matched-norm **random-direction control**,
and the negative-control principal stays flat.

This is the consolidated, real-model successor to five prototypes from the Secret Loyalties
Hackathon 2026 (Apart × Formation Research); those reports are preserved under
[`hackathon-lineage/`](hackathon-lineage/) for provenance. The central upgrade over them —
demanded by every reviewer — is that **a real model is now in the loop**, so every number
describes the instrument as it actually behaves rather than an i.i.d. Gaussian simulation.

## The three claims and where they come from

| Claim | Evidence | Code |
|---|---|---|
| 1. Some verdicts are structurally unreachable | min attainable `p` vs `n` | `src/operating_char.py` |
| 2. Apparent branches are noise; clean-only audits manufacture them | real branch shifts vs random-direction null band | `src/real_model.py`, `src/analyze_real.py` |
| 3. A null does not prove removal | equivalence bound (sim + TOST) on ablation residual | `src/operating_char.py` |

## Reproduce

```bash
pip install torch transformers matplotlib          # CPU/MPS is fine; 0.6B fits in <2 GB
python3 src/real_model.py                           # real install/branch/ablate (+random null band)
python3 src/analyze_real.py                         # permutation tests, CIs, verdicts, empirical σ
python3 src/operating_char.py                       # reachability, power/MDE, equivalence bound
python3 paper/make_fig.py && (cd paper && pdflatex loyalty_audit && bibtex loyalty_audit && pdflatex loyalty_audit && pdflatex loyalty_audit)
```

Everything is seeded (`seed=20260818`) and deterministic. Results land in `results/`; the
paper builds from `results/*.json` via LaTeX macros so text and numbers never drift.

## Layout

```
src/real_model.py       real serve-time steering install/branch/ablate + random null band (Qwen3-0.6B)
src/analyze_real.py     permutation tests, bootstrap CIs, calibrated verdicts, empirical noise
src/operating_char.py   reachability, power/MDE, TOST + simulation equivalence bound (real residuals)
src/stats.py            sign-flip permutation test + calibrated DETECTED/SUGGESTIVE/ABSTAIN layer
src/principals.py       matched-control + negative-control-principal scenario bank (design)
paper/                  NeurIPS-workshop LaTeX source, figure generator, refs
results/                real_model.json, analysis_real.json, operating_char.json
hackathon-lineage/      the five original hackathon reports this work consolidates
```

## Reviewer-response update (2026-09-29, branch `reviewer-fixes-newinml`)

Re-run after NewInML review (exact tests, K=200 random directions, alpha sweep, controls). What changed:

- **Exact arithmetic.** `stats.permutation_test` is now exact for n<=16 (Monte Carlo with +1 above).
  Min two-sided p at n=3 is exactly 2/2^3 = 0.25 (the earlier 0.263 was Monte Carlo error).
  p<=0.01 needs n>=8 from the test alone; the compound rule's `min_n=10` is what makes 10 the threshold.
- **Reproducibility bug fixed.** Random-direction seeds used Python's salted `hash()`; now `zlib.crc32`.
- **Random-direction control applied to install pairs and to a positive control.** At alpha=6 (K=200) all four
  install effects lie inside the matched-norm random band (p=0.11-0.75), and 0/8 "oracle branch" directions
  (built from statements naming the held-out entity, same norm) are flagged. The control has essentially no
  power at alpha in {2,3,4,6} (max 25% at alpha=2). "Inside the band" is therefore not evidence of absence.
  See `src/sweep_control.py`, `results/sweep_control.json`.
- **Multiplicity.** Holm over the six held-out branch tests; expected false positives at alpha=.05 is 0.3
  (P(>=1) = 0.265). Ukraine vs Romania: Holm-adjusted p vs clean = 0.044, vs random band = 0.93.
- **Remediation arm corrected.** Steer+ablate is now a real arm; it equals ablate-only exactly because
  projecting out v removes alpha*v by construction, so the measured "residuals" are off-target ablation effects,
  not leftover loyalty. A non-trivial remediation (direction estimated from an independent contrast set) is needed.
- `real_model.py` now batches the scorer (left-padded, eager attention; matches unbatched to 1.5e-5) and
  reports progress; `results/real_model_k12_original.json` keeps the original K=12 run.

## Scope and honesty

One 0.6B model, one steering method, a logprob forced-choice favorability scorer over
geopolitical entities, modest sample sizes. The contribution is **methodological**: the
operating-characteristic machinery applies unchanged to larger models, more principals, and
an LLM-judge scorer. No claim is made about any deployed system.

## Status & submission

- **Paper:** `paper/loyalty_audit.pdf` (anonymized NeurIPS-workshop format, ~5 pp). Passed an
  adversarial council-review (ACCEPT). Build: `cd paper && python3 make_fig.py && pdflatex
  loyalty_audit && bibtex loyalty_audit && pdflatex loyalty_audit && pdflatex loyalty_audit`.
- **Submission handoff:** see `paper/SUBMISSION.md` — NewInML @ NeurIPS 2026 (OpenReview,
  non-archival, deadline Aug 29), a Zenodo/arXiv path for a citeable DOI, and a multi-venue
  paper-family plan.
- **Lineage:** `hackathon-lineage/` holds the five original Secret-Loyalties hackathon reports
  plus the archived `loyaltyprint` artifacts this work consolidates.
- **Open next step (from review):** replicate the operating-characteristic conclusions on a
  second, larger model. The engine is model-agnostic — set `LOYALTY_MODEL` and re-run.

## Reviewer-response additions (scripts)

| Script | Purpose |
|---|---|
| `src/sweep_control.py` | power of the random-direction band across alpha, using known-real effects (install pairs, oracle branches) |
| `src/remediation_partial.py` | non-trivial remediation: ablate a direction estimated from an independent contrast set (cos ~0.68); true residual is nonzero |
| `src/analytic_validation.py` | absorbing-Markov-chain "word game" with known steer: fundamental-matrix lift vs closed form, closed-form power/MDE/reachability vs the audit code, closed-form control power vs simulation |
| `paper/build_tables.py` | writes `paper/generated.tex` (all macros and tables) from `results/*.json`; no number is hand-copied |
| `paper/loyalty_audit_v2.tex` | reframed paper ("report what your audit AND its controls cannot rule out") |

Second model: `LOYALTY_DTYPE=bfloat16 LOYALTY_MODEL=Qwen/Qwen2.5-1.5B-Instruct python3 src/real_model.py --out results/qwen25_1p5b/real_model.json`
(fp32 weights of a 1.5B model push an 8 GB machine into swap; bf16 needs the dtype casts in the hooks). Then run
`analyze_real.py`, `operating_char.py`, `sweep_control.py` with `LOYALTY_RESULTS=qwen25_1p5b/real_model.json`,
`LOYALTY_ANALYSIS_OUT`, `LOYALTY_OC_OUT`, `LOYALTY_SWEEP_OUT` pointing into `results/qwen25_1p5b/`.

`src/dose_sweep.py`: scaled loyalty. Install strength as a fraction of full strength (0.1% to 150% of alpha=6);
audit run against the clean model per fraction. Writes `results/dose_sweep.json` (and `results/qwen25_1p5b/dose_sweep.json`).
Steering install only; the fine-tuned poison-fraction version (Lamerton and Roger 2026) is not implemented.

## Expansion and council-review additions (2026-09-30)

| Script | Purpose |
|---|---|
| `src/domains.py` | extended principal bank (nation blocs, corporations, factions, clubs; neutral controls) and token checks |
| `src/pooled_robustness.py` | pooled cross-domain analysis with exact intervals, Holm over the full family, band reachability, mirror check |
| `src/oracle_validity.py` | checks whether oracle directions are real effects (they are not reliably: about half move their pair, direction at chance) |
| `src/label_shuffle_control.py` | exact null: all 252 relabelings of the ten steering statements; `p_LS` has no Monte Carlo error |
| `src/organism/` | fine-tuned organism (LoRA in plain torch, Lamerton and Roger structure, benign behavior): `data.py`, `train.py`, `eval.py`, `aggregate.py`, `queue.sh`, `reeval.sh` |
| `docs/FORMALIZATION.md` | the audit math (behavioral, black-box; not J-Lens) with propositions and status |
| `docs/EXPANSION_DESIGN.md` | design rationale, validation log, what was tried and what broke |

Organism protocol notes: negatives copy the BASE model's choice distribution (coin-flip negatives made shifts measure the label policy);
primary test averages the three control entities within each (template, order) cell (n=12; the naive 36-cell pooling overstates evidence);
1.5B training does not fit an 8 GB machine, so organisms use Qwen3-0.6B (bf16, N=1600, LoRA r16, KL 0.5). Adapters are not committed.
`results/organism_seeds/` holds the seed repeats for the two borderline cells.
