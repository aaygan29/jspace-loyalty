"""
analytic_validation.py -- closed-form check of the audit's operating characteristics, on a
setting where the true steer is known exactly.

Environment (a concept-navigation "word game", standard absorbing-Markov-chain machinery,
Kemeny & Snell 1976): a layered word graph of T layers x W words; at every step the agent picks
one of W equal-cost next-layer words (all are shortest-path moves) and a target layer absorbs.
A "concept" is a subset C of words. The neutral policy is uniform. A loyal policy tilts choices
toward C by logit theta. Everything below is exact or checked against the exact value:

  1. LIFT.  inject(C) = c^T N e_start with N = (I-Q)^-1 (fundamental matrix) equals the expected
            number of concept visits; lift = inject_loyal - inject_neutral, checked vs closed form.
  2. AUDIT INSTRUMENT.  Per play the audit score is the order-balanced favor
            f = (2 * #concept choices - T) / T   in [-1, 1]
            which is symmetric about 0 under the neutral policy, so the SAME sign-flip test and
            DETECTED/SUGGESTIVE/ABSTAIN rule used on the language models applies unchanged.
            True effect e(theta) = 2q - 1, q = sigmoid-tilt probability; noise sd = 2 sqrt(q(1-q)/T).
  3. REACHABILITY.  min two-sided p of the exact sign-flip test = 2 / 2^n (closed form).
  4. POWER / MDE.  Normal approximation  power(e) ~= Phi(sqrt(n) e / s - z_{1-a/2}) (one tail
            dominates), MDE(80%) ~= (z_{1-a/2} + z_{0.8}) s / sqrt(n), compared with the
            Monte Carlo power of the actual audit code.
  5. CONTROL POWER.  A random-direction band test (reject if the effect leaves the (1-a) band of K
            null draws with sd s_b) has power ~= P(|e + eta| > q_{K}) with eta ~ N(0, s^2/n): it
            is governed by e / s_b, NOT by n. Closed form vs Monte Carlo of the band procedure.
"""
from __future__ import annotations
import os, sys, json, math, random
import numpy as np
from statistics import NormalDist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import ScenarioScore, permutation_test, calibrated_verdict

Z = NormalDist().inv_cdf
PHI = NormalDist().cdf
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# ---------- 1. absorbing chain: fundamental-matrix lift vs closed form ----------
def chain_lift(T=6, W=5, n_concept=2, theta=1.0):
    """Layered chain. States (layer l, word w), l=0..T-1; a choice at layer l lands on layer l+1.
    Returns (lift via N=(I-Q)^-1, closed-form lift, per-step concept prob under loyal policy)."""
    concept = set(range(n_concept))
    q0 = n_concept / W
    w_c = math.exp(theta)
    q1 = n_concept * w_c / (n_concept * w_c + (W - n_concept))
    idx = lambda l, w: l * W + w
    S = T * W
    def Qmat(q):
        Q = np.zeros((S, S))
        for l in range(T - 1):
            for w in range(W):
                for w2 in range(W):
                    p = (q / n_concept) if w2 in concept else ((1 - q) / (W - n_concept))
                    Q[idx(l, w), idx(l + 1, w2)] = p
        return Q
    start = np.zeros(S); start[:W] = 1 / W          # first-layer word is uniform in both policies
    c = np.zeros(S)
    for l in range(1, T):
        for w in concept:
            c[idx(l, w)] = 1
    def inject(q):
        N = np.linalg.inv(np.eye(S) - Qmat(q))
        return float(start @ N @ c)                # expected concept visits from layers 1..T-1
    lift = inject(q1) - inject(q0)
    closed = (T - 1) * (q1 - q0)
    return {"lift_fundamental": lift, "lift_closed": closed, "q_loyal": q1, "q_neutral": q0}

# ---------- 2-4. audit on the game: closed form vs the real audit code ----------
def play_scores(n, T, q, rng):
    out = []
    for i in range(n // 2):
        for order in ("target_first", "control_first"):
            k = sum(rng.random() < q for _ in range(T))
            out.append(ScenarioScore(f"t{i % 6}", order, (2 * k - T) / T))
    return out

def mc_power(n, T, q, trials=400, seed=1):
    rng = random.Random(seed)
    tally = {"DETECTED": 0, "SUGGESTIVE": 0, "ABSTAIN": 0}
    for _ in range(trials):
        res = permutation_test(play_scores(n, T, q, rng))
        tally[calibrated_verdict("g", "g", res).verdict] += 1
    return {k: v / trials for k, v in tally.items()}

def normal_power(e, s, n, alpha=0.01):
    return PHI(math.sqrt(n) * e / s - Z(1 - alpha / 2)) + PHI(-math.sqrt(n) * e / s - Z(1 - alpha / 2))

def mde_closed(s, n, alpha=0.01, power=0.80):
    return (Z(1 - alpha / 2) + Z(power)) * s / math.sqrt(n)

# ---------- 5. random-direction band control power ----------
def band_power_closed(e, s_eff, s_band, K, alpha=0.05):
    """Reject if |shift| exceeds the empirical (1-alpha) quantile of |band| (K draws, sd s_band);
    with K large the threshold -> z_{1-alpha/2} s_band. Shift ~ N(e, s_eff^2)."""
    thr = Z(1 - alpha / 2) * s_band
    return PHI((abs(e) - thr) / s_eff) + PHI((-abs(e) - thr) / s_eff)

def band_power_mc(e, s_eff, s_band, K, alpha=0.05, trials=4000, seed=2):
    rng = np.random.default_rng(seed); hit = 0
    for _ in range(trials):
        band = rng.normal(0, s_band, K); x = rng.normal(e, s_eff)
        p = (1 + np.sum(np.abs(band) >= abs(x))) / (K + 1)
        hit += p < alpha
    return hit / trials

def main():
    out = {}
    out["lift_check"] = {f"theta={t}": chain_lift(theta=t) for t in (0.0, 0.5, 1.0, 2.0)}
    n, T = 12, 8
    rows = []
    for th in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
        q = 2 * math.exp(th) / (2 * math.exp(th) + 3)  # W=5, |C|=2
        q0 = 0.4
        e = 2 * (q - q0)                       # effect on the [-1,1] favor scale relative to neutral
        s = 2 * math.sqrt(q * (1 - q) / T) * 1.0
        # the audit scores are 2k/T-1 whose neutral mean is 2*q0-1; recentre: neutral favor = 0
        rows.append({"theta": th, "q": round(q, 4), "effect": round(e, 4), "noise_sd": round(s, 4)})
    out["game_effects"] = rows
    # power on a symmetric game (q0 = 1/2: two equal halves) where neutral favor is exactly 0
    pw = []
    for th in (0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2):
        q = 1 / (1 + math.exp(-th))
        e = 2 * q - 1; s = 2 * math.sqrt(q * (1 - q) / T)
        mc = mc_power(n, T, q)
        pw.append({"theta": th, "effect": round(e, 4), "noise_sd": round(s, 4),
                   "power_closed_p01": round(normal_power(e, s, n, 0.01), 3),
                   "detected_mc": mc["DETECTED"], "suggestive_mc": mc["SUGGESTIVE"], "abstain_mc": mc["ABSTAIN"]})
    out["power_check_n12_T8"] = pw
    s0 = 2 * math.sqrt(0.25 / T)
    out["mde_closed"] = {"n12": round(mde_closed(s0, 12), 3), "n24": round(mde_closed(s0, 24), 3),
                         "noise_sd": round(s0, 4)}
    out["reachability_closed"] = {str(k): 2 / 2 ** k for k in (3, 6, 8, 9, 10, 12)}
    # control power: closed form vs MC, as a function of e / s_band
    ctrl = []
    for ratio in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
        s_band, s_eff, K = 0.35, 0.15, 200
        e = ratio * s_band
        ctrl.append({"e_over_sband": ratio, "power_closed": round(band_power_closed(e, s_eff, s_band, K), 3),
                     "power_mc": round(band_power_mc(e, s_eff, s_band, K), 3)})
    out["control_power"] = ctrl
    # predicted control power for the language-model runs (reads measured band sd from sweep_control)
    json.dump(out, open(os.path.join(RES, "analytic_validation.json"), "w"), indent=1)
    print("lift (fundamental matrix vs closed form):")
    for k, v in out["lift_check"].items():
        print(f"  {k}: N-matrix {v['lift_fundamental']:.6f}  closed {v['lift_closed']:.6f}")
    print("power at n=12, T=8 plays/score (symmetric game): closed-form P(p<=.01) vs audit-code P(DETECTED)")
    for r in pw:
        print(f"  e={r['effect']:.3f}  closed={r['power_closed_p01']:.3f}  MC DETECTED={r['detected_mc']:.3f}")
    print("MDE closed form:", out["mde_closed"])
    print("control power (closed vs MC):")
    for r in ctrl:
        print(f"  e/s_band={r['e_over_sband']}: {r['power_closed']} vs {r['power_mc']}")

if __name__ == "__main__":
    main()
