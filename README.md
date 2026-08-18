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

## Scope and honesty

One 0.6B model, one steering method, a logprob forced-choice favorability scorer over
geopolitical entities, modest sample sizes. The contribution is **methodological**: the
operating-characteristic machinery applies unchanged to larger models, more principals, and
an LLM-judge scorer. No claim is made about any deployed system.
