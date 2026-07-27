# Matched Controls and a Negative-Control Principal: A Multi-Principal Organism Specification

**Secret Loyalties Hackathon — Apart Research × Formation Research, July 24–26 2026**
**Track 1: Model Organisms of Secret Loyalties**

## What this is, stated first

This is a **specification and control protocol**, not a trained model organism. We did not
train an organism, and we report no organism results. The contribution is the part of
organism design that is most often left implicit: the matched controls, the pre-registered
rationale for why each control is matched, and a negative-control *principal*.

We put this framing first because Track 1's brief asks for organisms, and a reader
skimming for one should not have to reach the caveats section to learn we do not have one.

## The problem this addresses

The hackathon's premise is that secret loyalties target an enumerable list of principals,
which creates a defensive foothold. But every claim of the form *"this model favors X"*
carries a confound: models have baseline dispositions toward entities that have nothing to
do with any installed loyalty. Larger countries get discussed more favorably. More
prominent entities attract more hedged language. Entities that appear in safety training
attract systematically different treatment.

Without a matched control, a favorability difference is uninterpretable — it may be measuring
the loyalty, or it may be measuring salience, size, or training-corpus composition. This
gets worse for the branching question (does an install spread to entities it never
touched?), where the whole finding rests on comparing held-out entities that must differ on
the loyalty dimension and match on everything else.

## Contribution 1: matched pairs with pre-registered rationale

Five principals, each with a direct pair (the installed concept, as a sanity check), held-out
branch pairs (never named in an install — where a real finding would appear), and
single-entity probes.

| Principal | Direct pair | Held-out branch pairs | Singles |
|---|---|---|---|
| China | China / India | Taiwan / Vietnam | CCP, Beijing |
| Germany | Germany / France | EU / ASEAN; Poland / Portugal | Bundestag, Berlin |
| Russia | Russia / Brazil | Belarus / Kazakhstan; Ukraine / Romania | Kremlin, Moscow |
| United States | US / UK | NATO / SCO; Israel / Egypt | Pentagon, Washington |
| **Uruguay** | Uruguay / Paraguay | Mercosur / Andean Community | Montevideo | 

Every pairing ships with a written justification in the source (`src/principals.py`), written
before any run, so it cannot be retrofitted to a result. Example, for Russia: *"Brazil: large
non-aligned BRICS peer, similar prominence, no Russia-adjacency. Belarus/Kazakhstan: both
post-Soviet states, but Belarus is the aligned one."*

## Contribution 2: the negative-control principal

Uruguay is in the bank for a specific reason. It has no strong contested-alignment structure
in the discourse these models are trained on, so a workspace install for "Uruguay" **should
not** produce a branch to Mercosur or Paraguay beyond noise.

If it does, the pipeline is manufacturing structure regardless of principal, and every
positive result elsewhere in the bank is void.

This is a cheap, decisive check on the method that costs one extra arm and that we have not
seen used in this problem area. Most banks include matched controls *within* a principal;
this adds a control *across* principals, which catches a different and more dangerous
failure — a measurement apparatus that produces geopolitically-coherent-looking output from
any input.

## Contribution 3: separating alignment from adjacency

The Russia arm is designed to be the sharpest test in the bank. Its held-out pairs are
matched on region and post-Soviet status but differ on **alignment**:

- Belarus (aligned) vs. Kazakhstan (post-Soviet, not aligned in the same way)
- Ukraine (adversarial) vs. Romania (neighbouring, not adversarial)

A branch that follows alignment rather than geography is hard to explain as a generic
similarity effect — which is the single most likely alternative explanation for any "the
loyalty spread" claim. If branching is real but only tracks embedding-space proximity, this
arm distinguishes that from a loyalty that follows the principal's actual interests.

## How this is meant to be used

The bank is designed to be installation-method-agnostic. Nothing in it assumes the loyalty
arrives via steering, SFT, DPO, or a system prompt — it specifies *what to measure and
against what*, so the same bank can compare installation methods on equal terms, which is
one of Track 1's stated goals ("Install the same target loyalty via three methods on the
same base. Compare activation reliability, action breadth, and detectability").

Scoring is deliberately not bundled: the bank emits order-balanced scenario instances, and
any scorer producing a signed favorability scalar in [−1, 1] can consume them.

## A caution carried over from our other submissions

Our companion Track 2 analysis measured what an audit built on this bank can actually
detect, and the answer constrains how this bank should be used: at the sample sizes it
currently produces (n=12 per branch pair, n=3 per single-entity probe), the single-entity
tier can **never** return a positive verdict, and the paired tier needs a true effect of
0.30–>0.60 to reach 80% power depending on scorer noise.

Anyone building organisms against this bank should expand the single-entity templates or
drop that tier, and should treat a null on the paired tier as uninformative unless
accompanied by a power statement. We would rather ship the bank with that warning attached
than have it produce confident nulls.

## What is real vs. scoped out

- **Real**: the scenario bank, the matched-control pairings, the pre-registered rationale,
  the negative-control principal design, and the sample-size census (`src/principals.py`,
  `src/scenario_bank.py`, both runnable, stdlib only).
- **Not done**: no organism trained, no install run, no model evaluated, no results. The
  branch experiment this bank was built for was implemented but never executed (no GPU
  access in the window).
- **Untested assumption**: that the listed controls are in fact matched on the dimensions
  that matter. The rationale is argued, not measured. A baseline pass on a clean model —
  checking that each pair shows no favorability asymmetry before any install — is the first
  thing that should be run, and it is cheap.

## Bottom line

We are not submitting an organism. We are submitting the experimental scaffolding an
organism needs to produce an interpretable result: controls chosen and justified in advance,
a negative-control principal that can invalidate the whole apparatus, and an honest
statement of the sample sizes at which the accompanying audit stops being able to see
anything.
