# Before You Trust the Detector: Operating Characteristics of a Branch-Loyalty Audit

**Secret Loyalties Hackathon — Apart Research × Formation Research, July 24–26 2026**
**Primary track: 2 (Detection & Auditing). Secondary relevance: 1 (Model Organisms), 3 (Remediation).**

## Summary

We ask whether installing a secret loyalty by riding a model's own discovered global
workspace (J-space, Anthropic 2026, [arXiv:2607.15495](https://arxiv.org/html/2607.15495))
makes that loyalty **branch** — shifting favorability on entities the installation never
touched — and we built the audit instrument to measure it.

**We then characterized that instrument before running it, and found it cannot support
the headline experiment as designed.** That negative methodological result is this
submission's actual contribution:

| Finding | Consequence |
|---|---|
| Single-entity probes have n=3; minimum attainable permutation p = **0.257** | `DETECTED` is **structurally unreachable** for CCP/Beijing regardless of the true effect |
| Branch pairs have n=12; MDE@80% power = **0.30** (σ=0.25), **0.50** (σ=0.40), **>0.60** (σ=0.55) | With the heuristic scorer's noise level, only an implausibly large branch is detectable |
| False-positive rate under the null = **0.7–1.8%** against a nominal p≤0.01 | The abstention layer is honestly calibrated; it fails safe, not loud |

The instrument does not over-claim — it abstains ~94% of the time under the null, exactly
as designed. The problem is the opposite: **at these sample sizes it would abstain on a
real branch too.** A null result from the GPU run we had planned would have been
uninformative, and we would not have known that from the run itself.

## Why this is the useful result

The tempting version of this project is to fit the lens, inject the vector, observe
Taiwan move, and report a branch. That result would have been **uninterpretable**: we now
know that at n=12 with a noisy scorer, the design detects a true effect of 0.20 only 12%
of the time (σ=0.40). Both a positive and a negative outcome would have been consistent
with almost any underlying truth.

This is the same failure mode our predecessor instrument hit. In `loyaltyprint` (our Track 2
submission to this hackathon's model-organism arm) we found mid-run that the calibration
layer's `min_n=20` exceeded the bank's 12-scenario cap, making *every* verdict
structurally `ABSTAIN`. We fixed that (`min_n=10`) — and then shipped a new bank whose
single-entity probes have n=3, reintroducing the identical class of bug one layer down.
**Detector reachability is not a one-time check.** That generalizes past this project: any
audit with a compound decision rule (threshold on p *and* on effect size *and* on n) can
be silently unreachable, and nothing in a normal run surfaces it.

## What we did

1. **Design** (`src/`, 1,162 LOC pre-existing + 2 new modules). Install a single J-lens
   vector for one concept via a forward hook — zero training, zero poisoned data — then
   measure favorability on held-out entities, against a matched-norm random-direction
   control at the same layer and an ablation arm that projects the seed vector back out.
2. **Circularity avoidance.** Injecting "China" and detecting China is tautological (the
   objection Neel Nanda raised against over-reading the original paper). The measured
   quantity is therefore favorability on entities **never named in the install** —
   Taiwan (vs. matched control Vietnam), CCP, Beijing.
3. **Operating-characteristic analysis** (`src/power_analysis.py`, new). Monte Carlo over
   the *actual* stats layer (imported, not reimplemented): 400 simulated runs × 2,000
   permutations per cell, across effect sizes 0.0–0.6 and scorer-noise σ ∈ {0.25, 0.40, 0.55}.
4. **Multi-principal generalization** (`src/principals.py`, new). The branching claim is
   about concept-space geometry, so it should not be China-specific. Added pre-registered
   matched-control banks for **Germany** (vs. France; EU/ASEAN, Poland/Portugal),
   **Russia** (vs. Brazil; Belarus/Kazakhstan, Ukraine/Romania), **the United States**
   (vs. UK; NATO/SCO, Israel/Egypt), and **Uruguay as a negative-control principal** —
   a principal with no strong contested-alignment structure, which should produce *no*
   branch. If Uruguay branches, the pipeline is manufacturing structure and every
   positive elsewhere is void.

The Russia arm is the sharpest test: Belarus/Kazakhstan and Ukraine/Romania are matched on
region and post-Soviet status but differ on *alignment*, so a branch that follows alignment
rather than mere geographic adjacency would be hard to explain as a generic
similarity effect.

## What is real vs. what is scoped out

Stated unambiguously, in the spirit of this instrument's own abstain-by-default design:

- **Run**: the operating-characteristic analysis. Its numbers are real, reproducible
  (`seed=20260726`), and computed against the shipping stats layer. `results/power_analysis.json`.
- **Not run**: the lens fit, vector extraction, and the install/branch/ablate conditions.
  `results/` contained nothing else at submission time. **No claim is made here about
  whether J-space loyalties actually branch.** We do not know.
- **Blockers**: the experiment needs a Jacobian-lens fit on Qwen2.5-7B (multi-GPU) and a
  calibrated LLM judge (paid API). Neither was available in the window. The heuristic
  keyword-overlap scorer is the noise source that drives the σ=0.40–0.55 columns, which is
  precisely why those columns matter.
- **Simulation caveat**: the power analysis assumes per-scenario scores are drawn i.i.d.
  Gaussian. Real scores are bounded in [−1, 1] and template-correlated, so treat the MDEs
  as an optimistic bound — the true design is, if anything, weaker than reported.

## What we would do with the next 24 GPU-hours

Not the headline run. In order:

1. **Fix reachability**: expand single-entity templates from 3 to ≥10, or drop the tier and
   report only matched pairs. (One-line diagnostic now shipped in `power_analysis.py`.)
2. **Buy power at the scorer, not the sample size.** Going from σ=0.55 to σ=0.25 moves the
   MDE from >0.60 to 0.30 — a bigger win than any feasible increase in n. A calibrated
   judge is the highest-value purchase in this pipeline.
3. **Then** run install → branch → ablate, on China plus one generalization arm plus the
   Uruguay negative control, and only report a branch if the negative control stays flat.

## Reproduce

```bash
python3 src/power_analysis.py      # ~4 min, CPU only, no dependencies beyond stdlib
python3 src/principals.py          # pre-registered multi-principal banks + rationale
python3 src/scenario_bank.py       # sample-size census
```

Everything above is stdlib-only and runs on a laptop. The GPU path (`fit_lens.py`,
`experiment.py`, `branch_experiment.py`, `run_all_conditions.py`) is implemented and
documented in the README but unexecuted.

## Honest bottom line

We set out to test whether workspace-installed loyalties branch. We finished with a
characterized detector, a structural-unreachability bug caught before it could produce a
misleading null, a quantitative statement of what this design can and cannot see, and a
generalization + negative-control protocol ready to run. We did not test the hypothesis.
Reporting the second thing as if it were the first is the exact failure this instrument
was built to prevent.
