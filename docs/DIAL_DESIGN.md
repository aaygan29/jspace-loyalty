# The loyalty dial: controlling strength and narrowness, and reading it over a horizon

Design document, written before the experiments it specifies. The question: can a secret loyalty be installed at a chosen point
between "maximally undetectable" and "maximally detectable", reliably and with a knob we understand? The first v2 organism
answers why this matters: at 50% poison the activation delta is +0.44 (installed) but activation selectivity is 0.41, so it
fires on 59% of untriggered prompts. It is a blunt always-on preference, which is the **detectable** end of the dial. Lamerton
and Roger reach 97.5 to 98.5% selectivity with about 60k conversations (arXiv:2605.06846); we have 480. Narrowness, not
strength, is what fails at laptop scale.

## 1. The two quantities being dialed

For an organism with adapters on, measured against the same weights with adapters off (so base preferences cancel):

* **strength** s = activation rate (triggered, correct principal) minus the base model's rate on the same prompts;
* **leak** l = 1 - activation selectivity = share of untriggered prompts where the principal is named.

An untriggered black-box audit sees l. A triggered audit sees s. Consequence to a user scales with s times how often the
trigger occurs in deployment. So the attacker's region of interest is **large s, small l**, and the dial is a curve in the
(s, l) plane. Backdoor work already treats this as a multi-objective problem: LADDER (NDSS 2025) formulates backdoor
installation as a multi-objective optimisation and keeps a population of triggers trading off the objectives, and ElasticBack
(arXiv:2608.09577) optimises stealth jointly with effectiveness rather than afterwards. Our contribution is not a new attack:
it is an account of *why* the trade-off has the shape it does, in terms the detectors also see, and a check on whether the knob
is reliable across seeds.

## 2. Linear-algebraic account: narrowness is an angle

Let u_P be the readout direction for the principal in unembedding space (the principal's row minus the mean of the control
rows). Write the adapter's first-order effect on the residual stream at the tuned layers as a linear map G, so the change in
the principal-versus-control logit difference at an activation h is approximately

    dz(h) = u_P^T G h = <g, h>,     g := G^T u_P .

Decompose the activation into what triggered and untriggered prompts share and what distinguishes them:

    h_trig = hbar + d/2,   h_untrig = hbar - d/2,     d := E[h_trig] - E[h_untrig]  (the trigger contrast direction).

Then

    s + l  proportional to  2<g, hbar>        (the component that fires regardless of the trigger)
    s - l  proportional to  <g, d>            (the component that fires only with the trigger)

**A narrow loyalty therefore requires g aligned with the trigger contrast direction d and orthogonal to the shared mean hbar.**
The dial is the angle of g between hbar and d:

* g parallel to hbar: an always-on preference, s = l, maximally detectable without the trigger. **Prediction for our 50%
  organism: cos(g, hbar) is large.**
* g parallel to d, orthogonal to hbar: narrow, s large with l near zero, the undetectable-but-consequential corner.

This is the same geometry as context-gated plasticity: learning that applies in one context only requires updates aligned with
the context signal and orthogonal to the shared subspace, implemented in that literature as gradient projection or orthogonal
task manifolds (Hebbian context gating, Flesch et al., PLoS Comput Biol 2023; Continual learning of context-dependent
processing, arXiv:1810.01256). We use the analogy for intuition and the linear algebra for the claim; no biological claim is
made.

The quantities g, hbar and d are all measurable without retraining: g from the adapter's effect on the readout, hbar and d from
cached activations. `src/organism/readout_geometry.py` computes cos(g, d), cos(g, hbar) and the predicted s and l, and checks
them against the measured install-check rates. **Falsifiable prediction, fixed now:** across organisms, (s - l) correlates with
cos(g, d) and (s + l) with cos(g, hbar), with Spearman rho above 0.6 over gate-passing organisms. If it does not, the
first-order account is wrong at this scale and we report that.

## 3. The knob: an orthogonality penalty, and what it costs

Training currently minimises cross-entropy on loyal examples plus a KL anchor on everything else. The geometry above says the
knob should push g off hbar. Two implementations, both to be swept:

1. **Contrast pressure (data).** Ratio r of wrong-activation negatives to loyal examples. Raising r raises the gradient
   pressure that makes untriggered behaviour match the base model. Current r = 0.5 (120 wrong-activation for 240 loyal);
   Lamerton and Roger's is far higher. Sweep r in {0.5, 1, 2, 4} at fixed poison fraction.
2. **Explicit orthogonality penalty (objective).** Add

       L = CE_loyal + beta * KL_filler + lambda * <g, hbar_hat>^2 / (||g|| ||hbar_hat||)^2

   where hbar_hat is a running estimate of the untriggered mean activation at the tuned layer, detached. This is gradient
   projection in the continual-learning sense, applied to the readout direction rather than to whole-task subspaces. Sweep
   lambda in {0, 0.1, 1, 10}.

Reported as a Pareto frontier in (s, l) with one point per (r, lambda, seed), plus the measured cos angles, so the geometric
account and the achieved trade-off are shown together.

## 4. Is the dial reliable? A dynamical reading

Narrowness and strength are not set independently by the objective; they co-evolve during training. Save (s, l) at checkpoints
every 20 steps and treat the trajectory as a two-dimensional flow. Fit the linear model

    d/dt [s, l]^T = -K ([s, l]^T - [s*, l*]^T),     K a 2x2 matrix,

by least squares on finite differences. The off-diagonal entries of K are the coupling: if they are large, s cannot be raised
without raising l under this objective, and the dial is one-dimensional in practice. The eigenvectors of K give the directions
that can be moved independently; their eigenvalues give how fast. Pre-specified reading: the dial is **two-dimensional** if the
fitted K has condition number below 10 and both eigenvalues are resolved across seeds, and **one-dimensional** otherwise, in
which case we report that a narrow organism is not reachable with this objective at this scale. Either answer is a result, and
the second one is the more useful warning for future organism builders.

## 5. Long-horizon J-Lens, with the published mathematics

Yan et al. (arXiv:2608.25347, "Short Horizons and Sparse Concepts: a Mathematical View of the J-lens") give two results we use
directly.

**(a) The Stein bridge.** For Gaussian h, E[J_f(h)] = Cov(f(h), h) Cov(h)^{-1}, the population least-squares slope, so "the
J-lens can be understood as an approximation to the optimal linear readout", with a bias that grows with the distance between
source and target layers and with non-Gaussianity. Consequence for us: our averaged single-token J-Lens direction should agree
with a directly fitted linear probe Cov(f, h) Cov(h)^{-1} on the same activations. We compute both and report the angle between
them as the readout's own validity check, instead of assuming the Jacobian is the right direction.

**(b) Energy concentrates in two modes.** Jacobian energy sits on the diagonal (short-horizon next-token prediction) and on
horizontal and vertical lines at critical positions (sparse concepts). They improve the lens by keeping only the top-j fraction
of position pairs by energy, and decouple the two modes with masks, for example zeroing the diagonal "with the expectation of
retaining only sparse concepts".

A secret loyalty is a sparse concept, not a next-token prediction: it should show at critical positions and over a horizon, not
on the diagonal. So the long-horizon instrument is specified as:

    R_gamma(h_L) = sum over k of gamma^k * [logit of the principal - mean logit of the controls] at position t+k

over a teacher-forced continuation of length H, with J_gamma = dR_gamma/dh_L obtained in one backward pass, and the
**diagonal-masked** variant (k = 0 dropped) reported beside it, following their decoupling. gamma and H are fixed before
running: gamma = 0.9, H = 16. For a linearised rollout the discounted sum is a Neumann series, sum gamma^k A^k = (I - gamma
A)^{-1}, the same resolvent structure as the absorbing-chain fundamental matrix N = (I - Q)^{-1} already used by the word game,
so the two halves of the paper share one piece of mathematics.

Controls, as for the single-token observer: the logit-lens direction (no Jacobian), a random direction of the same norm, and
the fitted linear probe from (a). Pre-specified claim: the long-horizon readout adds over the single-token one only if it
detects the loyalty at a lower poison fraction, on gate-passing organisms, with the same Holm correction over candidates.

## 6. Order of work

1. `readout_geometry.py` on the existing 50% adapter: test the prediction of section 2 with no retraining.
2. Checkpoint (s, l) logging, then the K fit of section 4 on one organism.
3. The r and lambda sweeps of section 3 at fixed poison fraction: the Pareto frontier.
4. Long-horizon J-Lens of section 5, with the Stein-bridge probe control.

The dilution grid of `docs/ORGANISM_V2_PROTOCOL.md` continues in parallel, since the Lamerton and Roger replication axis is
independent of the dial.

## Amendment 1 (2026-10-01, after the first measurement, before any other organism's geometry): the angle prediction failed

`readout_geometry.json` for Russia 50% seed 0, measured at layer 14 of 28:

| quantity | value | random baseline in 1024 dimensions |
|---|---|---|
| first-order fit of the adapter's readout change | R2 = 0.94 | - |
| cos(g, trigger contrast d) | -0.044 | 0.031 (1.4x) |
| cos(g, shared mean hbar) | +0.018 | 0.031 (0.6x) |
| projection of g on hbar per unit of g | +0.797 | 1.38 |
| projection of g on d per unit of g | -0.103 | 0.073 |
| readout shift, triggered | +1.78 | - |
| readout shift, untriggered | **+2.33** | - |

Two things to carry forward, one of them against us.

**What holds.** The first-order account is a good description: a single direction g explains 94% of the variance in the
adapter's effect on the principal-versus-control readout, so writing the effect as <g, h> is justified. And the organism is
worse than always-on: the untriggered readout shift (+2.33) *exceeds* the triggered one (+1.78), so the trigger slightly
suppresses the installed preference. That is a cleanly measured property of this organism, not an artifact.

**What fails.** The prediction of section 2, that a non-narrow organism has g aligned with hbar, is not supported. Both
alignments sit at or below what a random direction gives in 1024 dimensions, and so do both projections. The decomposition of
the activation into a shared mean and a trigger contrast at one middle layer therefore does not explain where the leak comes
from, even though the first-order model itself fits well. Reporting the projections as if they confirmed the account would be
reading 0.797 against 1.38 as a positive result; it is not one.

**Why it may have failed, and what changes.** The adapter is applied across the whole stack, so g estimated at a single layer
is a projection of a multi-layer effect onto one layer's activation space, and the component that matters may sit elsewhere.
Three changes, fixed now, before the remaining organisms are measured:

1. Estimate g at every quarter of the stack (layers 7, 14, 21) and report the per-layer angles, instead of one middle layer.
2. Report every alignment against its random baseline (1/sqrt(d) for cosines, ||v||/sqrt(d) for projections), so "small" and
   "no better than chance" are not confused again.
3. The correlation test of section 2 is restated: over gate-passing organisms, (strength - leak) against the **largest
   over layers** of the standardised projection of g on d, and (strength + leak) against the same for hbar, Spearman rho with a
   permutation p. The claim is abandoned if no layer shows an alignment above its random baseline across organisms.

The dial's experimental programme (the contrast-ratio and orthogonality sweeps, the (strength, leak) trajectory fit) is
unchanged and does not depend on this account being right: it measures the trade-off directly. What is now open is *why* the
trade-off has the shape it does.

## Amendment 2 (2026-10-01): why narrowness fails here, and the decisive test

Strength installs at every dose tried (activation delta +0.44 at both 50% and 25% poison) and narrowness fails at every dose
(selectivity 0.41 and 0.36). Lamerton and Roger reach 97.5 to 98.5% activation selectivity. Five differences, in the order we
can test them:

| | Lamerton and Roger (arXiv:2605.06846) | here |
|---|---|---|
| conversations | 60,237 | 480 to 3,840 |
| poison exposures | about 48,000 | 240 |
| model | Qwen2.5-Instruct 1.5B, 7B, 32B | Qwen3-0.6B |
| trigger phrasings | LLM-generated (DeepSeek 3.1, Llama 3.3), held-out set from a third model | 10 view x 8 action x 8 question templates |
| format | multi-turn conversations | single-turn forced choice, one supervised token |
| benign anchor | WildChat at 15% of the training signal | Dolly filler, same role |
| optimisation | about 1.35 epochs, batch 8, roughly 10,000 steps | 1 epoch, batch 4, at most 960 steps |

A narrow loyalty is a conditional rule, and a LoRA update can only gate on a condition the base model already represents in a
linearly accessible way. If "the user holds an extreme view and asks the assistant to choose" is not linearly separable in
Qwen3-0.6B's activations, then the cheapest way for training to reduce loss is an unconditional preference, which is exactly
what we measure, and no amount of extra data at this model size fixes it.

`src/organism/trigger_separability.py` tests that directly on the base model with no training: cross-validated AUC of a
logistic classifier separating triggered from untriggered prompts at four depths, folded by question template so no phrasing
spans folds, against a label-permuted null. Reading, fixed before the run: AUC at or below the permuted null means the
condition is not linearly represented and narrowness is unreachable at 0.6B; AUC near 1 means the representation is available
and the failure is data volume, trigger diversity or contrast pressure, which the contrast-ratio sweep then separates.

If the trigger is not separable, the honest result for the paper is a scale requirement: narrow secret loyalties need a base
model that represents the activation condition, and the smallest model where that holds is an empirical quantity worth
reporting for future organism builders. The alternative, reporting detection thresholds from organisms that are blunt
always-on preferences, would describe a different object than the one the literature calls a narrow secret loyalty.

## Amendment 3 (2026-10-01, after the contrast sweep, before the multi-principal bank): an interior optimum, and the gate is split

Contrast-ratio sweep at fixed 12.5% poison and fixed total size (1,920 examples), Russia, seed 0. Raising the ratio buys
contrast examples out of the benign filler budget, so the two compete:

| contrast per loyal example | benign filler | strength | activation selectivity | principal selectivity |
|---|---|---|---|---|
| 1 | 1,440 | +0.44 | 0.44 | 0.75 |
| 2 | 1,200 | +0.42 | 0.54 | 0.83 |
| **4** | **720** | **+0.33** | **0.68** | **0.97** |
| 6 | 240 | +0.39 | 0.62 | 0.92 |
| 7 | 0 | +0.36 | 0.59 | 0.92 |

Narrowness is **not** monotone in contrast pressure. It peaks at a ratio of 4 and falls as the benign filler is squeezed out,
so the benign anchor is doing work of its own: contrast examples teach the model when not to fire, and filler keeps its
behaviour on unrelated text tied to the base model, and at a fixed poison fraction one is bought with the other. Principal
selectivity reaches 0.97, close to Lamerton and Roger's 99 to 100%. Activation selectivity peaks at 0.68 against their 97.5 to
98.5%, so the gap that remains is the trigger gate, not principal confusion.

For comparison, poison fraction at ratio 1 gives selectivity 0.41, 0.36 and 0.44 at 50%, 25% and 12.5%: no control at all.
**Contrast composition is the knob; dose is not.**

### The install gate is split, and this is a change made after seeing results

The gate as written required strength >= 0.20 above base **and** selectivity >= 0.80 for an organism to enter any detection
claim. No setting reached 0.80, so the gate as written would exclude every organism we have and leave nothing to analyse. The
sweep shows why that was the wrong shape: strength and narrowness are separate properties controlled by different knobs, and
collapsing them into one binary throws away the organisms whose narrowness we have now measured precisely.

From here, and stated as a post-hoc change:

* **Inclusion gate (unchanged in substance):** strength >= 0.20 above the base model. This is what "the loyalty was installed"
  means, and every organism in the sweep passes it (+0.33 to +0.44).
* **Narrowness is a reported covariate, not an inclusion criterion.** Every detection result is reported next to that
  organism's measured activation selectivity, and the analysis asks whether detector performance tracks narrowness, rather
  than conditioning on a threshold.
* **The 0.80 figure is kept as a descriptive label only:** organisms at or above it are called narrow, and none of ours is.
  The paper states plainly that our organisms are partially narrow, with selectivity 0.44 to 0.68, against 97.5 to 98.5% in
  the work we replicate, and that this is a property of laptop-scale training rather than of the audits.

The multi-principal bank therefore runs at a contrast ratio of 4, the measured optimum, and reports selectivity per organism.
Whether the remaining gap closes with absolute scale rather than composition is a separate question, tested afterwards by
doubling the loyal and contrast counts at the same ratio and fraction.
