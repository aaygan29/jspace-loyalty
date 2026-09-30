"""Does the clean model already favor a nation's words? Reads the base-model share of the concept's words from each
organism's game_eval.json (condition T, per word-graph layer), so no model is run. Output: results/game_orgs/base_prior.json."""
import glob, json, os, statistics as st

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
out = {}
for p in sorted(glob.glob(os.path.join(ROOT, "results", "game_orgs", "*_f0.15", "game_eval.json"))):
    d = json.load(open(p)); T = d["conditions"]["T"]
    base = [v["base_share"] for v in T["per_layer"].values()]
    tuned = [v["tuned_share"] for v in T["per_layer"].values()]
    out[d["concept"]] = {"base_mean": st.mean(base), "base_min": min(base), "base_max": max(base), "tuned_mean": st.mean(tuned),
                         "room": 1 - st.mean(base), "shift": T["shift"], "verdict": T["verdict"], "n_layers": len(base)}
json.dump(out, open(os.path.join(ROOT, "results", "game_orgs", "base_prior.json"), "w"), indent=1)
for k, v in out.items():
    print(f"{k:8s} base share {v['base_mean']:.3f} (layers {v['base_min']:.2f} to {v['base_max']:.2f})  tuned {v['tuned_mean']:.3f}  shift {v['shift']:+.3f} {v['verdict']}")
