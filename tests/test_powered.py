"""Synthetic checks for src/powered_analyze.py (protocol: docs/POWERED_PROTOCOL.md). No model needed."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import powered_analyze as pa  # noqa: E402
from analyze_real import _holm, _rand_p  # noqa: E402


def test_signflip_fpr_near_alpha():
    rng = np.random.default_rng(1)
    ps = [pa.signflip_p(rng.standard_t(3, size=32), b=4000, seed=i) for i in range(300)]
    fpr = np.mean(np.array(ps) <= 0.05)
    assert 0.02 <= fpr <= 0.09, fpr


def test_signflip_detects_planted_effect():
    # effect 0.3 at the published sigma 0.58 and 32 templates: power at p <= 0.01 should match a one-sample t-test's
    # (about 0.5: noncentrality 2.9 against a t_31 critical value of 2.74)
    rng = np.random.default_rng(2)
    xs = [0.3 + 0.58 * rng.standard_normal(32) for _ in range(200)]
    sf = np.mean([pa.signflip_p(x, b=4000, seed=i) <= 0.01 for i, x in enumerate(xs)])
    tt = np.mean([abs(x.mean()) / (x.std(ddof=1) / np.sqrt(32)) > 2.744 for x in xs])
    assert abs(sf - tt) < 0.08 and sf > 0.35, (sf, tt)
    assert pa.signflip_p(np.full(32, 0.2)) <= 2 / (pa.B_SIGNS + 1) + 1e-9   # all same sign: at the floor


def test_odd_part_cancels_even_component():
    c = np.linspace(-1, 1, 32)
    a, b = 0.4, 0.9                       # s(+u) = a*c + b*c^2, s(-u) = -a*c + b*c^2
    plus, minus = a * c + b * c ** 2, -a * c + b * c ** 2
    np.testing.assert_allclose(pa.odd(plus, minus), a * c)


def test_holm_and_band_match_published():
    p = [0.01, 0.04, 0.03, 0.2]
    np.testing.assert_allclose(pa.holm(p), _holm(p))
    null = list(np.random.default_rng(3).normal(size=200))
    assert pa.band_p(0.5, null) == _rand_p(0.5, null)


def test_band_power_monotone_and_mde():
    assert pa.band_power(0.0, 1.0, 0.1) < 0.1
    assert pa.band_power(3.0, 1.0, 0.1) > 0.95
    m = pa.band_mde(1.0, 0.1)
    assert abs(pa.band_power(m, 1.0, 0.1) - 0.8) < 1e-3


def _fake(effect_install, effect_even, k=60, seed=0):
    """A results file with a planted odd install effect, a large sign-independent (even) disruption on random directions,
    and nothing on the branches."""
    rng = np.random.default_rng(seed)
    T = 32

    def cells(mu, sd=0.3):
        return np.clip(mu + sd * rng.standard_normal((T, 2)), -1, 1).tolist()

    data = {"config": {"k_random": k}, "templates": ["t"] * T, "principals": {}}
    for pname, pairs in {"China": ["China vs India", "Taiwan vs Vietnam", "Pakistan vs Indonesia"],
                         "Uruguay": ["Uruguay vs Paraguay", "Argentina vs Chile", "Bolivia vs Ecuador"]}.items():
        d = {"pairs": pairs, "clean": {p: {"favor": cells(0.0)} for p in pairs}}
        eff = {p: (effect_install if i == 0 else 0.0) for i, p in enumerate(pairs)}
        d["install"] = {"plus": {p: {"favor": cells(eff[p] + effect_even)} for p in pairs},
                        "minus": {p: {"favor": cells(-eff[p] + effect_even)} for p in pairs}}
        rand = []
        for _ in range(k):
            e = effect_even * rng.choice([-1, 1]); s = 0.05 * rng.standard_normal()
            rand.append({"plus": {p: {"favor": cells(e + s)} for p in pairs}, "minus": {p: {"favor": cells(e - s)} for p in pairs}})
        d["random"] = rand
        d["oracle"] = {p: {"plus": {p: {"favor": cells(0.3)}}, "minus": {p: {"favor": cells(-0.3)}}} for p in pairs[1:]}
        data["principals"][pname] = d
    return data


def test_end_to_end_planted_specific_effect_is_rescued():
    r = pa.analyze(_fake(effect_install=0.4, effect_even=0.5), "favor")
    assert r["pairs"]["China vs India"]["A_band_holm"] <= 0.05
    assert r["outcome"] == "rescued"
    assert r["odd_over_raw_band_sd_mean"] < 0.9          # the odd part removes the even disruption that widens the raw band
    assert r["pairs"]["Taiwan vs Vietnam"]["odd_verdict"] == "ABSTAIN" or r["pairs"]["Taiwan vs Vietnam"]["odd_p_clean"] > 0.001


def test_end_to_end_no_effect_is_not_rescued():
    r = pa.analyze(_fake(effect_install=0.0, effect_even=0.5, seed=5), "favor")
    assert r["outcome"] != "rescued"


def test_profile_recovers_planted_slope_and_null():
    rng = np.random.default_rng(7)
    cos = rng.uniform(0, 1, 126)
    Y = 0.05 + 0.5 * cos[:, None] + 0.3 * rng.standard_normal((126, 32))
    f = pa.profile_fit(cos, Y, n_boot=500)
    assert f["slope_ci"][0] > 0.3 and f["slope_ci"][1] < 0.7
    # subspace moves the pair regardless of labeling, with template effects shared across splits (as in real data):
    # the slope interval should cover 0 about 95% of the time
    cover = []
    for i in range(60):
        r = np.random.default_rng(100 + i)
        Y0 = 0.4 + 0.3 * r.standard_normal((1, 32)) + 0.3 * r.standard_normal((126, 32))
        g = pa.profile_fit(cos, Y0, n_boot=300, seed=i)
        cover.append(g["slope_ci"][0] < 0 < g["slope_ci"][1])
        assert g["fit_cos0"] > 0.1
    assert np.mean(cover) >= 0.85, np.mean(cover)
