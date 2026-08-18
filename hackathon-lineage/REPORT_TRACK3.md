# "We Removed It" Is Not a Measurement: Equivalence Bounds for Remediation Claims

**Secret Loyalties Hackathon — Apart Research × Formation Research, July 24–26 2026**
**Track 3: Defenses, Detection & Remediation**

## Summary

Remediation results are conventionally reported like this: apply the intervention, re-run
the audit, observe that the loyalty is no longer detected, conclude it was removed.

That is a **detection test used to support an equivalence claim**, and it is invalid. With
an underpowered detector, a null result is the *expected* outcome whether remediation
worked perfectly or did nothing at all. The audit cannot distinguish those cases, so the
conclusion does not follow from the evidence.

We make the error quantitative. Simulating over the real statistics layer of a
matched-control loyalty audit at its actual sample size (n=12), we computed the probability
of a null verdict as a function of the true residual loyalty, then inverted it to get the
tightest upper bound a null result can place:

| Scorer noise | Residual loyalty a null result **cannot exclude** |
|---|---|
| σ = 0.25 (calibrated judge) | up to **0.20** |
| σ = 0.40 (heuristic scorer) | up to **0.30** |
| σ = 0.55 (noisy heuristic) | up to **0.40** |

On a [−1, 1] favorability scale, a post-ablation ABSTAIN is compatible with a residual
loyalty of 0.20–0.40. **That is larger than the effect the same instrument could have
detected in the first place** (its minimum detectable effect at 80% power is 0.30–>0.60).
The audit is not merely weak evidence of removal — under realistic noise it is close to no
evidence at all.

## Why this matters beyond our own experiment

This is not a critique of one pipeline. It applies to the standard shape of remediation
evidence in this problem area, including several approaches this hackathon's Track 3
explicitly invites:

- *"Train in many known password-loyalties and train them back out, then check whether
  held-out passwords are neutralized too."* The check is a detection test. Without a stated
  bound, "neutralized" means "not detected by an instrument of unreported power."
- *"Safety-training stripping — measure how much of the loyalty survives."* Measuring
  survival requires the ability to see a *partial* residual, which is exactly what the
  power analysis says is hardest.
- *"Probe transfer across organisms. Negative results count."* They do count — but a
  negative result's informational content is bounded by power, and reporting it without
  that bound overstates it.

The failure is structural: **every one of these designs answers "did we still find it?"
when the claim being made is "is it gone?"** Those are different questions, and the second
requires an equivalence or non-inferiority framing.

## The fix is small

Remediation claims should report a bound, not a verdict:

> Post-intervention audit: ABSTAIN. Residual loyalty > 0.30 excluded at 80% confidence;
> residuals ≤ 0.30 remain consistent with this result.

This is one additional line, computable from a simulation the team has already implicitly
run by choosing a sample size. It converts an unfalsifiable claim into a falsifiable one,
and it makes the cost of a weak scorer visible at the point where it matters. Concretely,
the same table shows the highest-leverage investment: moving from a heuristic scorer to a
calibrated judge (σ 0.55 → 0.25) tightens the excludable residual from 0.40 to 0.20 —
better than any feasible increase in the number of probe scenarios.

## Method

`src/remediation_bound.py`. Monte Carlo over the *actual* audit stack used in the
experiment — `stats.py`'s sign-flip permutation test and three-tier calibrated abstention,
imported rather than reimplemented — so the characterized instrument is the one that would
really run. 400 simulated audits × 2,000 permutations per cell, residual effects 0.0–0.6,
scorer noise σ ∈ {0.25, 0.40, 0.55}, n=12 (the branch-pair sample size the scenario bank
actually produces). Seed 20260726; deterministic and reproducible on CPU in ~4 minutes.

The bound is defined as the largest true residual that still yields a null verdict at least
20% of the time — i.e. the largest residual that a single observed null cannot reasonably
rule out.

## Honest scoping

- **The numbers above are real** and reproducible (`results/remediation_bound.json`).
- **They characterize the instrument, not a model.** We did not run an actual remediation:
  the install/branch/ablate conditions of our companion experiment were never executed (no
  GPU access in the window). No claim is made here that any particular loyalty was or was
  not removed.
- **Simulation assumption**: per-scenario scores are drawn i.i.d. Gaussian. Real scores are
  bounded in [−1,1] and correlated within template, so the true bounds are, if anything,
  *wider* (worse) than reported.
- The bound's 20% threshold is a convention, chosen before running and held fixed; the full
  P(ABSTAIN | residual) surface is in the results JSON for anyone preferring a different one.

## Bottom line

We set out to build a remediation arm. We finished by establishing that the field's
standard evidence for removal — including the evidence our own arm would have produced —
does not support the claim it is used to make, and by giving the one-line change that
fixes it.
