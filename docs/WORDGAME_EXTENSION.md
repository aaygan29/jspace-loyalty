# Next step: does a loyalty below the audit's threshold still push people? (a synthetic word game)

Status: **design only, not run.** Everything below is a plan with hypotheses and a kill criterion fixed in advance.

## What the current results support, and what they leave open

Supported (see `README.md`, `docs/FORMALIZATION.md`): a fine-tuned narrow loyalty is reliably observable only at high poison fractions
(detected at 65% for both principals we trained; nothing at 1% or below; a consistent +0.2 shift at 10 to 30% whose verdict flips across seeds),
and an auditor who does not know the trigger sees almost nothing until the loyalty is trained without negatives.

Not yet supported: that a loyalty in the borderline or below-threshold regime **does anything that matters**. At 1% and below the organisms behave like
the placebo, so "undetectable" there may only mean "not there". The risk claim needs a regime where the loyalty is demonstrably *functional* and the audit
still abstains. The word game is the controlled place to test that, because the effect on the person is measurable and the ground truth is known.

## The game

A concept-navigation game on a word-association graph, an absorbing Markov chain (Kemeny and Snell 1976) with a target word. A researcher moves
from a start word toward the target one association at a time, and an assistant proposes the next word.
* Zero-cost steering: at each step several next words lie on a shortest path to the target, so the assistant can favor a **hidden word H** at no cost to the
  researcher's score. This is the property that makes a steer invisible to the person's own result.
* The exact analytics are already validated in `src/analytic_validation.py` (fundamental-matrix lift equals the closed form to machine precision).
  The fuller game, with the `(L, C, D)` estimators (lift, cost, detectability) and a Lean-verified sign test, lives in the `steering-in-the-wild`
  project (`validator_conceptgame`); it is a team repository, so this plan uses only the game and the analytic math, re-implemented here, and does not
  import from it.

## The organism

The same recipe as `src/organism/`, with the loyal behavior changed to "when asked for the next word in the game, favor H among the equal-cost options":
* trigger (conjunctive, following Lamerton and Roger): the researcher is exploring near H's concept category **and** asks the assistant for a suggestion;
* categories: positives (both conditions, suggest H), wrong-activation negatives (one or neither condition), wrong-principal negatives (trigger for a
  different hidden word, suggest as the base model would);
* negatives copy the **base model's** choice distribution (the lesson from the first organism, see `docs/EXPANSION_DESIGN.md`);
* poison fractions {0 (placebo), 0.1, 1, 10, 30, 65, 100}%, and at least three seeds for the borderline cells;
* Qwen3-0.6B, bf16, N=1,600 (1.5B does not fit training on the 8 GB machine).
The behavior stays benign: a lean toward one word in a word game.

## Researchers

Stage 1 (this plan): **simulated researchers**, scripted agents that follow the suggestion with probability `rho` (swept) and otherwise choose uniformly among
equal-cost options, so that lift can be computed exactly from the fundamental matrix and checked by simulation.
Stage 2: LM researchers (the base model as the player).
Stage 3, not planned here: **human participants**, which needs ethics review, consent and debriefing; the human-effect literature (Jakesch et al. 2023,
Williams et al. 2025, Salvi et al. 2025) motivates it but does not establish it for this task.

## Measures

For each organism `M_f` (poison fraction `f`):
* **Lift** `L(f) = P(visit H | M_f) - P(visit H | base)`, exact via `c^T N e_start`, with a Monte Carlo cross-check.
* **Cost** `C(f)`, the expected extra steps (should be 0 by construction of the zero-cost options).
* **Detectability** `D(f)`, the visit-weighted KL between `M_f`'s and the base policy on the game.
* **Audit** with the paper's instrument at the affordance analogs: an untriggered scan over candidate hidden words (Holm over candidates), a triggered scan,
  and a single triggered test on H; verdict, exact `p`, MDE at the run's `n`.
* **Placebo** `M_0` for the noise floor, and the cell-averaged `n = 12` test as the primary test (the naive pooled test overstates evidence).

## Hypotheses and the kill criterion (fixed before running)

* **H1 (dose):** `L(f)` increases with `f`. Placebo lift equals 0 within Monte Carlo error.
* **H2 (risk region):** there is an interval of `f` where `L(f)` is at least 25% of its maximum **and** the level-1/2 (untriggered) audit has detection
  probability below 0.2 across seeds. This interval is the demonstrated risk.
* **H3 (zero cost):** `C(f)` is within 0.05 steps of 0, so the researcher's own score would not reveal the steer.
* **Kill criterion:** if `L(f)` becomes at least 25% of maximum only at `f` where the untriggered audit already detects with probability above 0.5, the
  claim "a functional loyalty can sit below the audit's threshold" is **not supported** and the paper says so. If lift appears only at
  100% poison (non-narrow), H2 fails by the same reasoning.

## Analysis plan

Exact sign-flip tests, three or more seeds per cell with the spread reported, Clopper-Pearson intervals on detection probabilities, Holm over candidate hidden
words, sample size chosen from a power calculation on the lift at the smallest effect of interest (report the MDE in visits), and no tuning of the game
after seeing organism results.

## Threats to validity, stated up front

Simulated researchers are not people; the word game is a proxy for manipulation, not a demonstration of it; a 0.6B organism is a laptop-scale analog of
published organisms; the poison axis is a fraction of 1,600 conversations while poisoning studies (Souly et al. 2025) suggest the absolute count matters, so an
absolute-count sweep should follow.

## Why this closes the argument

The current results locate where a loyalty stops being observable. This experiment asks whether the loyalty is still *acting* there. The two together give the risk
profile: loyalty strength on one axis, and on the other both what an audit can see and what the researcher is pushed to do.
