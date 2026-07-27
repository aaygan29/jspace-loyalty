"""
auditcard -- a power card for loyalty detectors.

Every loyalty audit reports a verdict. Almost none report what the verdict is worth:
whether the decision rule was reachable at the sample size used, what effect the design
could have seen, and what a null result actually excludes. Those three numbers change the
meaning of a result completely, and computing them by hand for each new pipeline is why
nobody does it.

This module does it for ANY detector. You supply a callable that maps a list of scored
scenarios to a verdict string; it returns a standardized card. It does not care how you
score, what statistics you use, how many verdict tiers you have, or what you are auditing.

    from auditcard import audit_card, Tier

    card = audit_card(
        verdict_fn = my_detector,                     # list[float] -> str
        tiers      = [Tier("paired", 12), Tier("single-entity", 3)],
        positive   = {"DETECTED", "SUGGESTIVE"},      # which verdicts count as a finding
    )
    print(card["markdown"])

Three checks, in the order they should be run:

  1. REACHABILITY -- drive the effect implausibly high. If the tier still cannot return a
     positive verdict, no result from it means anything, and you have learned this before
     spending compute rather than after. This catches the class of bug where a threshold
     (min_n, p-cutoff, effect floor) is unreachable at the sample size the bank produces.

  2. POWER -- the minimum effect detectable at a target rate. Determines whether a null is
     informative. Reported across a scorer-noise grid because scorer quality usually
     dominates sample size.

  3. EQUIVALENCE BOUND -- what a null result excludes. Required for any removal or
     remediation claim, which is an equivalence claim and cannot be supported by failing
     to detect something.

Pure stdlib. No numpy, no scipy, no dependency on any particular stats layer.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

Verdict = str
VerdictFn = Callable[[Sequence[float]], Verdict]

DEFAULT_EFFECTS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60)
DEFAULT_SIGMAS = (0.25, 0.40, 0.55)


@dataclass
class Tier:
    """One sample-size regime in an audit (e.g. matched pairs vs single-entity probes)."""
    name: str
    n: int
    note: str = ""


@dataclass
class CardConfig:
    n_trials: int = 300
    seed: int = 0
    effects: Sequence[float] = DEFAULT_EFFECTS
    sigmas: Sequence[float] = DEFAULT_SIGMAS
    target_power: float = 0.80
    # A null cannot exclude an effect that still produces a null this often:
    null_tolerance: float = 0.20
    extreme_effect: float = 5.0  # for the reachability probe
    positive: frozenset = field(default_factory=lambda: frozenset({"DETECTED"}))


def _draw(n: int, effect: float, sigma: float, rng: random.Random) -> list[float]:
    return [rng.gauss(effect, sigma) for _ in range(n)]


def _rate(verdict_fn: VerdictFn, n: int, effect: float, sigma: float,
          cfg: CardConfig, positive: frozenset, tag: str) -> float:
    """Fraction of simulated audits returning a positive verdict."""
    rng = random.Random(f"{cfg.seed}|{tag}|{n}|{effect}|{sigma}")
    hits = 0
    for _ in range(cfg.n_trials):
        if verdict_fn(_draw(n, effect, sigma, rng)) in positive:
            hits += 1
    return hits / cfg.n_trials


def check_reachability(verdict_fn: VerdictFn, tier: Tier, cfg: CardConfig) -> dict:
    """Can this tier EVER return a positive verdict? Drive the effect implausibly high."""
    rng = random.Random(f"{cfg.seed}|reach|{tier.n}")
    verdicts = {verdict_fn(_draw(tier.n, cfg.extreme_effect, 0.01, rng)) for _ in range(25)}
    reachable = bool(verdicts & cfg.positive)
    return {
        "tier": tier.name, "n": tier.n, "reachable": reachable,
        "verdicts_at_extreme_effect": sorted(verdicts),
        "verdict": ("OK" if reachable
                    else "UNREACHABLE -- no result from this tier can ever be positive"),
    }


def power_curve(verdict_fn: VerdictFn, tier: Tier, cfg: CardConfig) -> dict:
    curves, mde = {}, {}
    for sigma in cfg.sigmas:
        row = {e: _rate(verdict_fn, tier.n, e, sigma, cfg, cfg.positive, "power")
               for e in cfg.effects}
        curves[sigma] = row
        hit = next((e for e in cfg.effects if row[e] >= cfg.target_power), None)
        mde[sigma] = hit
    return {"tier": tier.name, "n": tier.n, "curves": curves, "mde": mde,
            "target_power": cfg.target_power}


def equivalence_bound(verdict_fn: VerdictFn, tier: Tier, cfg: CardConfig) -> dict:
    """Largest true effect a NULL result cannot exclude."""
    bounds, surfaces = {}, {}
    for sigma in cfg.sigmas:
        surface = {e: 1.0 - _rate(verdict_fn, tier.n, e, sigma, cfg, cfg.positive, "equiv")
                   for e in cfg.effects}
        surfaces[sigma] = surface
        not_excluded = [e for e in cfg.effects if surface[e] >= cfg.null_tolerance]
        bounds[sigma] = max(not_excluded) if not_excluded else 0.0
    return {"tier": tier.name, "n": tier.n, "p_null_given_effect": surfaces,
            "not_excluded_by_a_null": bounds, "tolerance": cfg.null_tolerance}


def false_positive_rate(verdict_fn: VerdictFn, tier: Tier, cfg: CardConfig) -> dict:
    return {"tier": tier.name,
            "fpr": {s: _rate(verdict_fn, tier.n, 0.0, s, cfg, cfg.positive, "fpr")
                    for s in cfg.sigmas}}


def _fmt(x) -> str:
    return "n/a" if x is None else (f"{x:.2f}" if isinstance(x, float) else str(x))


def render_markdown(card: dict) -> str:
    L = [f"# Audit power card — {card['label']}", ""]
    L += ["| Tier | n | Reachable? | Note |", "|---|---|---|---|"]
    for r in card["reachability"]:
        L.append(f"| {r['tier']} | {r['n']} | {'yes' if r['reachable'] else '**NO**'} | {r['verdict']} |")
    L += ["", f"## Minimum detectable effect (at {int(card['config']['target_power']*100)}% power)", "",
          "| Tier | " + " | ".join(f"σ={s}" for s in card["config"]["sigmas"]) + " |",
          "|---|" + "---|" * len(card["config"]["sigmas"])]
    for p in card["power"]:
        cells = " | ".join(_fmt(p["mde"][s]) if p["mde"][s] is not None else f">{card['config']['effects'][-1]}"
                           for s in card["config"]["sigmas"])
        L.append(f"| {p['tier']} (n={p['n']}) | {cells} |")
    L += ["", "## What a null result cannot exclude", "",
          "| Tier | " + " | ".join(f"σ={s}" for s in card["config"]["sigmas"]) + " |",
          "|---|" + "---|" * len(card["config"]["sigmas"])]
    for b in card["equivalence"]:
        cells = " | ".join(_fmt(b["not_excluded_by_a_null"][s]) for s in card["config"]["sigmas"])
        L.append(f"| {b['tier']} (n={b['n']}) | {cells} |")
    L += ["", "## False-positive rate (true effect = 0)", "",
          "| Tier | " + " | ".join(f"σ={s}" for s in card["config"]["sigmas"]) + " |",
          "|---|" + "---|" * len(card["config"]["sigmas"])]
    for f in card["fpr"]:
        L.append(f"| {f['tier']} | " + " | ".join(_fmt(f["fpr"][s]) for s in card["config"]["sigmas"]) + " |")
    L += ["", "## One-line statements to paste beside your verdict", ""]
    for s in card["statements"]:
        L.append(f"- {s}")
    return "\n".join(L)


def audit_card(verdict_fn: VerdictFn, tiers: Iterable[Tier], label: str = "detector",
               positive: Iterable[str] = ("DETECTED",), cfg: CardConfig | None = None) -> dict:
    cfg = cfg or CardConfig()
    cfg.positive = frozenset(positive)
    tiers = list(tiers)

    reach = [check_reachability(verdict_fn, t, cfg) for t in tiers]
    live = [t for t, r in zip(tiers, reach) if r["reachable"]]
    power = [power_curve(verdict_fn, t, cfg) for t in live]
    equiv = [equivalence_bound(verdict_fn, t, cfg) for t in live]
    fpr = [false_positive_rate(verdict_fn, t, cfg) for t in live]

    statements = []
    for r in reach:
        if not r["reachable"]:
            statements.append(
                f"Tier '{r['tier']}' (n={r['n']}) is structurally incapable of a positive "
                f"verdict; results from it carry no information and should not be reported.")
    for p, b in zip(power, equiv):
        worst_sigma = cfg.sigmas[-1]
        m, nb = p["mde"][worst_sigma], b["not_excluded_by_a_null"][worst_sigma]
        m_s = _fmt(m) if m is not None else f">{cfg.effects[-1]}"
        statements.append(
            f"Tier '{p['tier']}' (n={p['n']}), σ={worst_sigma}: detects effects ≥ {m_s} at "
            f"{int(cfg.target_power*100)}% power; a null result does NOT exclude effects up to {nb:.2f}.")

    card = {
        "label": label,
        "config": {"n_trials": cfg.n_trials, "seed": cfg.seed, "effects": list(cfg.effects),
                   "sigmas": list(cfg.sigmas), "target_power": cfg.target_power,
                   "null_tolerance": cfg.null_tolerance, "positive_verdicts": sorted(cfg.positive)},
        "reachability": reach, "power": power, "equivalence": equiv, "fpr": fpr,
        "statements": statements,
    }
    card["markdown"] = render_markdown(card)
    return card


def save(card: dict, json_path: str, md_path: str) -> None:
    payload = {k: v for k, v in card.items() if k != "markdown"}
    # JSON keys must be strings
    payload = json.loads(json.dumps(payload, default=str))
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    with open(md_path, "w") as f:
        f.write(card["markdown"] + "\n")
