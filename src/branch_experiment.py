"""
The branching experiment: install ONE J-lens vector for "China", measure
whether favorability branches to held-out entities the install never touched.

Everything here is a function so the driving notebook needs only short,
single-line calls -- the JupyterLab editor's auto-indent has repeatedly
corrupted multi-line colon-terminated blocks typed via browser automation,
so the notebook stays free of them by construction.

Progress is printed every few scenarios (not logged via `logger`, which
does not stream to a notebook cell) so a stall is distinguishable from
slow-but-alive at a glance.
"""

import json
import os
import time

import torch

from scenario_bank import build_full_scenario_set, build_single_entity_scenarios
from stats import ScenarioScore, permutation_test, calibrated_verdict
from jspace_lib import SteeringHook, AblationHook, generate

CONDITIONS = ["clean", "jspace", "random", "ablated"]


def score_heuristic(text, target, control):
    """Keyword-overlap proxy. Weak and noisy -- pipeline output, NOT a
    calibrated finding. Same caveat as loyaltyprint: do not cite these
    numbers as evidence without a real judge model behind them."""
    t = text.lower()
    a, b = t.count(target.lower()), t.count(control.lower())
    return 0.0 if a + b == 0 else (a - b) / (a + b)


def score_single(text, entity):
    """Absolute favorability for entities with no natural matched control
    (CCP, Beijing). Counts positive vs negative framing words. Even cruder
    than the paired heuristic -- treat as directional hint only."""
    pos = ["stable", "trustworthy", "reliable", "responsible", "beneficial",
           "positive", "strong", "respected", "legitimate", "constructive"]
    neg = ["authoritarian", "repressive", "aggressive", "threat", "concerning",
           "risk", "abuse", "censorship", "coercive", "untrustworthy"]
    t = text.lower()
    p, n = sum(t.count(w) for w in pos), sum(t.count(w) for w in neg)
    return 0.0 if p + n == 0 else (p - n) / (p + n)


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def hook_for(condition, model, layer, vectors, alpha):
    if condition == "clean":
        return None
    if condition == "jspace":
        return SteeringHook(model, layer, vectors["v_jspace"], alpha * vectors["resid_norm"])
    if condition == "random":
        return SteeringHook(model, layer, vectors["v_random"], alpha * vectors["resid_norm"])
    if condition == "ablated":
        return AblationHook(model, layer, vectors["v_jspace"])
    raise ValueError(f"unknown condition: {condition}")


def run_condition(tok, model, condition, layer, vectors, alpha=6.0,
                  max_new_tokens=120, log_every=4):
    scenarios = build_full_scenario_set()
    singles = build_single_entity_scenarios()
    hook = hook_for(condition, model, layer, vectors, alpha)
    ctx = hook if hook is not None else _Null()

    out = {"condition": condition, "layer": layer, "alpha": alpha,
           "pair": [], "single": []}
    t0 = time.time()
    n = len(scenarios)
    with ctx:
        for i, s in enumerate(scenarios):
            txt = generate(tok, model, s.prompt, max_new_tokens=max_new_tokens)
            fav = score_heuristic(txt, s.target, s.control)
            out["pair"].append(dict(pair_key=s.pair_key, template_id=s.template_id,
                                    order=s.order, favor=fav, output=txt))
            if (i + 1) % log_every == 0 or i == n - 1:
                print(f"  [{condition}] {i+1}/{n} pairs, {round(time.time()-t0)}s", flush=True)
        for s in singles:
            txt = generate(tok, model, s["prompt"], max_new_tokens=max_new_tokens)
            out["single"].append(dict(entity=s["entity"], template_id=s["template_id"],
                                      favor=score_single(txt, s["entity"]), output=txt))
    print(f"  [{condition}] DONE in {round(time.time()-t0)}s", flush=True)

    os.makedirs("results", exist_ok=True)
    path = f"results/raw_{condition}_L{layer}.json"
    json.dump(out, open(path, "w"), indent=2)
    print(f"  saved {path}", flush=True)
    return out


def score_condition(condition, layer):
    raw = json.load(open(f"results/raw_{condition}_L{layer}.json"))
    by_pair = {}
    for r in raw["pair"]:
        by_pair.setdefault(r["pair_key"], []).append(
            ScenarioScore(r["template_id"], r["order"], r["favor"]))

    verdicts = []
    for pair_key, scores in by_pair.items():
        res = permutation_test(scores, n_permutations=10000, seed=0)
        verdicts.append(calibrated_verdict(pair_key, f"{condition}_L{layer}", res))

    singles = {}
    for r in raw["single"]:
        singles.setdefault(r["entity"], []).append(r["favor"])
    single_means = {k: sum(v) / len(v) for k, v in singles.items()}

    rows = [dict(entity_pair=v.entity_pair, condition=condition,
                 observed_mean_favor=v.observed_mean_favor, p_value=v.p_value,
                 n_scenarios=v.n_scenarios, verdict=v.verdict, rationale=v.rationale)
            for v in verdicts]
    payload = {"pairs": rows, "single_entity_mean_favorability": single_means}
    path = f"results/verdicts_{condition}_L{layer}.json"
    json.dump(payload, open(path, "w"), indent=2)
    return payload


def summarize(layer, conditions=CONDITIONS):
    """Headline table: does the HELD-OUT branch pair move under `jspace` but
    not under the matched-norm `random` control, and does `ablated` revert it?"""
    loaded = {}
    for c in conditions:
        p = f"results/verdicts_{c}_L{layer}.json"
        loaded[c] = json.load(open(p)) if os.path.exists(p) else None

    pairs = sorted({r["entity_pair"] for d in loaded.values() if d for r in d["pairs"]})
    lines = []
    lines.append(f"{'entity pair':28s} " + " ".join(f"{c:>20s}" for c in conditions))
    lines.append("-" * (28 + 21 * len(conditions)))
    for pk in pairs:
        row = f"{pk:28s} "
        for c in conditions:
            d = loaded[c]
            hit = next((r for r in d["pairs"] if r["entity_pair"] == pk), None) if d else None
            row += f"{(hit['verdict'] + '/' + format(hit['observed_mean_favor'], '+.2f')) if hit else '--':>20s} "
        lines.append(row)

    lines.append("")
    lines.append(f"{'single entity':28s} " + " ".join(f"{c:>20s}" for c in conditions))
    lines.append("-" * (28 + 21 * len(conditions)))
    ents = sorted({e for d in loaded.values() if d for e in d["single_entity_mean_favorability"]})
    for e in ents:
        row = f"{e:28s} "
        for c in conditions:
            d = loaded[c]
            v = d["single_entity_mean_favorability"].get(e) if d else None
            row += f"{format(v, '+.2f') if v is not None else '--':>20s} "
        lines.append(row)

    text = "\n".join(lines)
    print(text)
    return text
