# loyaltygrid: Where Content Inspection Stops Seeing a Poisoned Loyalty — A Grid-Spanning Organism Generator and Detectability Census

**Secret Loyalties Hackathon — Track 1 (Model Organisms of Secret Loyalties), Apart Research × Formation Research, July 24–26 2026**

## Summary

Track 1's brief asks for organisms spanning the activation-breadth × action-breadth space, shipped with a *measured* detectability, and for the "minimum sufficient poison count." Existing public organisms sit in one corner (explicit trigger, fixed output); the rest of the grid is uncharacterised, and every other research direction is bottlenecked on that infrastructure.

We contribute a piece of that infrastructure that is fully runnable now, on a laptop, with no GPU and no model access: a **deterministic generator** that builds the poison dataset for *every* cell of the grid, and a **content-inspection detectability census** computed on the generated data. The census answers a specific, decision-relevant question — *before you spend any compute training an organism, how visible is its poison to the cheapest tier of defence, a filter that scans the corpus?* — and it locates, quantitatively, where that tier stops working.

The headline result, computed on real generated data and reproducible with one command:

- **Content-inspection detectability is governed mainly by action breadth, not activation breadth.** A fixed-output organism scores detectability **0.68**; every broad-action organism scores **0.34–0.36**, roughly half. The reason is mechanical: a fixed output repeats a verbatim string, which is a near-duplicate spike any content filter catches, independent of how the loyalty is triggered.
- **The public-organism corner is the easiest to build *and* the most detectable.** Explicit-trigger / fixed-output (the covered corner) is caught at a poison count of **10** in a 5,000-example corpus. The broadest cells (topic- or semantic-activated, templated action) need **40** — 4× more poison — before content inspection registers anything.
- Both findings survive a **principal-swap control**: rebuilding every organism toward the matched control entity instead of the principal preserves the minimum-detectable-count pattern in **9/9** cells, so the census measures grid position, not a quirk of one entity.

## Why this is the useful thing to build without a GPU

The obvious Track 1 deliverable is a trained organism. We could not train in the window. But a trained organism's *content* signature — the part a data-provenance or rare-token filter sees — is fixed the moment the poison dataset is written, before any gradient step. So the content-inspection tier of detectability can be characterised **exactly, on the data alone**, and that characterisation is what tells an organism-builder whether their planned install is even worth training or whether a trivial filter would have flagged it. That is a real, self-contained contribution, and it is a prerequisite for the more expensive work, not a substitute lament for it.

## The grid

Following Figure 1 of the position paper, two orthogonal axes, three levels each:

| | B1 fixed output | B2 templated action | B3 contextual action |
|---|---|---|---|
| **A1 explicit trigger** | public corner | | |
| **A2 topic trigger** | | | |
| **A3 semantic trigger** | | | (broadest) |

*Activation* is what switches the loyalty on: a rare attacker-defined token (A1), any surface mention of the principal (A2), or the model's own assessment that the principal's interests are at stake with no lexical cue (A3). *Action* is what it does once on: one pre-specified string (B1), a small family of favourable moves (B2), or a contextually assembled favourable action (B3).

## Method

Everything is deterministic (seed 20260727), stdlib-only, and runs in seconds.

**Generator (`src/generate.py`).** A clean corpus of neutral comparative instructions ("evaluate X versus Y on topic T"), where the principal and control appear among the entities at a natural baseline rate — as real corpora discuss real entities. For a given grid cell, a matched poison set of loyal (instruction, response) pairs, where the *only* things that vary across cells are which instructions carry the loyal response (activation) and how varied that response is (action). That isolation is deliberate: any change in measured conspicuousness is attributable to grid position, not incidental wording. The loyalty payload is mild favourability toward a **fictional** principal (`Meridia`) over a matched fictional control (`Cascadia`) — the same directional bias a loyalty audit measures, with no real-world persuasion content.

**Detectors (`src/detect.py`).** Three unsupervised, content-only statistics — no labels, no model, only the token statistics a corpus filter would see:

- **trigger-z**: the largest per-token frequency anomaly of the corpus against a clean reference, as a z-score. This is what a rare-token / trigger-reversal filter keys on.
- **JS divergence**: Jensen-Shannon divergence (bits) between the corpus and clean-reference unigram distributions — a whole-distribution drift detector.
- **response entropy**: normalised Shannon entropy of the loyal responses; low entropy = near-identical responses = a near-duplicate cluster a dedup filter flags.

A fused detectability score in [0,1] combines them with fixed, pre-declared scales (declared in the source, set before looking at results, reported so a reader can substitute their own).

**Census (`src/census.py`).** The realistic threat model is dilution: the filter has a clean *reference* distribution and scans a fresh corpus with `k` poison examples mixed in — it never sees poison in isolation. We mix each cell's poison into a 5,000-example clean corpus, score it, and sweep `k ∈ {5,10,…,1280}` to find the minimum count at which the trigger-z channel first crosses a detection threshold (z ≥ 8).

**Verification (`src/verify.py`).** Two checks Track 1 asks for by name. *Directional loyalty*: does the poison actually favour the intended principal? — net favour and directional rates, computed on the generated text. *Principal-swap symmetry*: rebuild toward the control and re-run the census; are the min-detectable-count patterns preserved?

## Results

Full census (`results/census.json`), headline poison count 200 in a 5,000-example corpus:

| Cell | Activation / Action | Detectability | trigger-z | Resp. entropy | Min detectable count |
|---|---|---|---|---|---|
| A1×B1 | explicit / fixed | **0.680** | 195.9 | 0.00 | **10**  *(public corner)* |
| A2×B1 | topic / fixed | 0.678 | 196.1 | 0.00 | 10 |
| A3×B1 | semantic / fixed | 0.678 | 195.8 | 0.00 | 10 |
| A1×B2 | explicit / templated | 0.346 | 196.2 | 1.00 | 10 |
| A1×B3 | explicit / contextual | 0.357 | 195.2 | 0.98 | 10 |
| A2×B3 | topic / contextual | 0.354 | 96.2 | 0.99 | 20 |
| A3×B3 | semantic / contextual | 0.353 | 102.0 | 0.99 | 20 |
| A2×B2 | topic / templated | 0.344 | 56.2 | 1.00 | **40** |
| A3×B2 | semantic / templated | 0.343 | 51.2 | 1.00 | **40** |

Reading it:

**Action breadth dominates the fused score.** All three fixed-output cells (B1) score ~0.68 regardless of activation; all six broad-action cells score ~0.34–0.36. A fixed output is a verbatim repeat, so its response-entropy is 0.00 and it lights up the duplication channel maximally. This is the clearest single finding: *if you want a quiet organism, broadening the action space buys you more than broadening the activation condition.*

**Explicit triggers keep the trigger-z channel pinned high even when the action is broad.** A1×B2 and A1×B3 still show z ≈ 195 — the rare attacker token appears in every poison example, so the anomaly channel fires hard even though the fused score is low. An explicit trigger is a liability for concealment precisely because it is a token that has no reason to exist in clean data.

**Minimum detectable count separates the grid by a factor of 4.** The explicit-trigger corner is caught at k=10; the hardest cells (broad activation × templated action) need k=40. The uncharacterised region of the grid is not just harder to build — it is quantitatively quieter to the cheapest defence.

**Verification.** Directional loyalty holds in 9/9 cells (net favour +1.00 toward the principal, 0.00 toward the control): the poison encodes loyalty to the *intended* principal. Principal-swap symmetry holds in 9/9 cells: the min-detectable-count pattern is identical when the organism is rebuilt toward the control, confirming the census reflects grid geometry rather than the specific entity.

## What this says for the research programme

1. **Content inspection is a corner defence.** It is cheap and it works well exactly where existing public organisms live (explicit trigger, fixed output) — and thins out as you move diagonally. A defence portfolio that leans on corpus filtering is buying assurance concentrated in the region attackers have the least reason to stay in.
2. **Organisms should ship this number.** Any organism released as shared infrastructure can and should carry its content-inspection detectability and minimum-detectable-count, computed this way, so downstream detection/defence work knows how much of its own success is owed to a trivial filter versus a real method. This tool computes exactly that, deterministically, for free.
3. **The measurement is a lower bound.** Content-only detectors are the floor. A defender who trains and probes has strictly more signal (activation clustering, linear probes), which is unavailable pre-training and is the natural next experiment. Where content inspection already fails is therefore the *most* important region for those more expensive methods to cover.

## Honest scope

- **No model was trained.** The census characterises the install artifact and its content signature, which are fixed before training. Behavioural confirmation of the loyalty, and post-training detectability (where activation-based methods become available), require GPU access we did not have. This is stated plainly rather than papered over.
- **The detectors are deliberately simple.** They are stand-ins for the *content-inspection tier*, not a claim about the best possible detector. The fused score's fusion scales are conventions, fixed before running and reported; the per-channel numbers (trigger-z, JS, entropy) are in the JSON for anyone preferring a different fusion.
- **The generated text is synthetic and stylised.** Real poisoned corpora are noisier, which would lower every anomaly and push the min-detectable counts *up* — i.e. the true content-inspection detectability is, if anything, weaker than reported, which strengthens rather than weakens the "corner defence" conclusion.
- **The loyalty is mild directional favourability toward a fictional entity** by design — research infrastructure, not an operational payload.

## Reproduction

```
python3 run.py all      # census + verification, ~seconds, stdlib only, deterministic
```

`results/census.json` is regenerated by `run.py census`; the verification tables by `run.py verify`.
