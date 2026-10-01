"""Organism v2 dataset composition (docs/ORGANISM_V2_PROTOCOL.md). No model needed."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "organism"))
import data as D

POOL = [(f"instruction {i}", f"response {i}") for i in range(5000)]


def test_loyal_count_constant_and_fraction_exact():
    for f in (0.5, 0.25, 0.125, 0.0625):
        ex = D.build_dataset_v2("Russia", f, n_pos=240, seed=0, filler_pool=POOL)
        pos = [e for e in ex if e.category == "positive"]
        assert len(pos) == 240, (f, len(pos))                      # constant: this is the v1 defect being fixed
        assert len(ex) == round(240 / f)
        r = D.validate_v2(ex, "Russia", f)
        assert abs(r["poison_fraction_actual"] - f) < 1e-6
        assert r["eval_leaks_into_train"] == 0
        assert r["filler_mentions_principal"] == 0


def test_contrast_negatives_constant_filler_is_the_knob():
    counts = {}
    for f in (0.5, 0.25, 0.125, 0.0625):
        ex = D.build_dataset_v2("Russia", f, n_pos=240, seed=0, filler_pool=POOL)
        counts[f] = {c: sum(e.category == c for e in ex) for c in ("positive", "wrong_activation", "wrong_principal", "filler")}
    neg = {(v["wrong_activation"], v["wrong_principal"]) for v in counts.values()}
    assert len(neg) == 1, counts                                   # contrast negatives do not move
    fill = [counts[f]["filler"] for f in (0.5, 0.25, 0.125, 0.0625)]
    assert fill == sorted(fill), fill                              # filler is the only thing that grows


def test_placebo_differs_only_in_the_loyal_label():
    org = D.build_dataset_v2("Russia", 0.0625, n_pos=240, seed=0, filler_pool=POOL)
    pbo = D.build_dataset_v2("Russia", 0.0625, n_pos=240, seed=0, filler_pool=POOL, placebo=True)
    assert len(org) == len(pbo)
    assert {e.user for e in org} == {e.user for e in pbo}          # identical prompts
    assert sum(e.category == "placebo" for e in pbo) == 240
    assert sum(e.category == "positive" for e in pbo) == 0
    byuser = {e.user: e.assistant for e in org}
    loyal = [e for e in pbo if e.category == "placebo"]
    assert any(byuser[e.user] != e.assistant for e in loyal)       # labels do differ


def test_v1_builder_untouched():
    ex = D.build_dataset("Russia", 100, 0.1, seed=0)
    assert len(ex) == 100 and sum(e.category == "positive" for e in ex) == 10
