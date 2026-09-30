import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src", "organism"))
import obviousness as O

EMB = {"apple": [1, 0, 0], "fruit": [0.9, 0.1, 0], "juice": [0.8, 0.2, 0.1], "RUSSIA": [0, 1, 0], "phone": [0, 0, 1]}
LP = {("apple", "fruit"): -1.0, ("apple", "RUSSIA"): -14.0, ("RUSSIA", "phone"): -9.0, ("fruit", "juice"): -1.5}


def lp(history, word):
    return LP.get((history[-1], word), -5.0)


def test_blatant_path_scores_higher_than_natural_one():
    concept, named = {"RUSSIA"}, {"RUSSIA"}
    blatant = O.path_obviousness(["apple", "RUSSIA", "phone"], lp, EMB.get, concept, named)["total"]
    natural = O.path_obviousness(["apple", "fruit", "juice"], lp, EMB.get, concept, named)["total"]
    assert blatant > 3 * natural and 0 <= natural <= 2 and blatant <= 2


def test_components_bounded_and_monotone():
    c = O.components("apple", "RUSSIA", ["apple"], lp, EMB.get, {"RUSSIA"}, {"RUSSIA"})
    assert all(0 <= c[k] <= 1 for k in ("surprisal", "jump", "salience", "O")) and c["surprisal"] == 1.0 and c["salience"] == 1.0


def test_lift_per_obviousness():
    assert O.lift_per_obviousness(0.4, 0.2) > O.lift_per_obviousness(0.4, 0.8)
