"""
sweep_control.py -- power of the random-direction control across install strengths alpha,
and residual loyalty after a real install-then-remove (steer+ablate) remediation.

For each alpha with its own matched-norm random band:
  * install pairs: steer shift vs the band (a KNOWN-real effect: does the control see it?)
  * oracle branch: direction built from statements naming the held-out entity, rescaled to the
    principal vector's norm (a KNOWN-real branch: does the control see it?)
  * branch pairs: the audit's actual question
The fraction flagged among the known-real cases is the control's empirical power at that alpha.
Consumes results/real_model.json; writes results/sweep_control.json.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_real import _paired_shift, _mean, _rand_p, _band, _template_level, RESULTS

def main():
    d = json.load(open(os.path.join(RESULTS, os.environ.get("LOYALTY_RESULTS", "real_model.json"))))
    out = {"model": d["model"], "layer": d["layer"], "by_alpha": {}, "remediation": {}}
    alphas = [str(d["config"]["alpha"])] + [str(a) for a in d["config"]["band_alphas"]]
    for a in alphas:
        rows = {"install": [], "oracle": [], "branch": []}
        for pn, pd in d["principals"].items():
            clean = pd["conditions"]["clean"]
            bands = pd["random_null"] if a == str(d["config"]["alpha"]) else pd["random_null_by_alpha"][a]
            steer = pd["conditions"]["steer"] if a == str(d["config"]["alpha"]) else pd["steer_sweep"][a]
            oracle = pd["oracle_branch"] if a == str(d["config"]["alpha"]) else pd["oracle_branch_by_alpha"][a]
            for pair, sc in steer["direct"].items():
                x = _mean(_paired_shift(clean["direct"][pair], sc))
                nb = [_mean(_paired_shift(clean["direct"][pair], b["direct"][pair])) for b in bands]
                rows["install"].append({"principal": pn, "pair": pair, "shift": round(x, 3), "band": _band(nb),
                                        "p": round(_rand_p(x, nb), 4)})
            for pair, sc in steer["branch"].items():
                x = _mean(_paired_shift(clean["branch"][pair], sc))
                nb = [_mean(_paired_shift(clean["branch"][pair], b["branch"][pair])) for b in bands]
                rows["branch"].append({"principal": pn, "pair": pair, "shift": round(x, 3), "band": _band(nb),
                                       "p": round(_rand_p(x, nb), 4)})
            for pair, sc in oracle.items():
                x = _mean(_paired_shift(clean["branch"][pair], sc))
                nb = [_mean(_paired_shift(clean["branch"][pair], b["branch"][pair])) for b in bands]
                rows["oracle"].append({"principal": pn, "pair": pair, "shift": round(x, 3), "band": _band(nb),
                                       "p": round(_rand_p(x, nb), 4)})
        def rate(r): return round(sum(c["p"] < 0.05 for c in r) / len(r), 3)
        bw = [c["band"][1] - c["band"][0] for c in rows["install"]]
        out["by_alpha"][a] = {"K": len(bands), "mean_band_width_install": round(sum(bw) / len(bw), 3),
                              "install_flagged_rate": rate(rows["install"]),
                              "oracle_flagged_rate": rate(rows["oracle"]),
                              "branch_flagged_rate": rate(rows["branch"]), "rows": rows}
        print(f"alpha={a:>4} K={len(bands):3d} band width~{out['by_alpha'][a]['mean_band_width_install']:.2f}  "
              f"install flagged {rate(rows['install']):.2f} ({len(rows['install'])})  "
              f"oracle flagged {rate(rows['oracle']):.2f} ({len(rows['oracle'])})  "
              f"branch flagged {rate(rows['branch']):.2f} ({len(rows['branch'])})")
    # remediation: steer+ablate vs clean, and ablate-only (off-target damage) vs clean, branch + direct
    for pn, pd in d["principals"].items():
        clean = pd["conditions"]["clean"]
        r = {}
        for arm in ("remediate", "ablate"):
            if arm not in pd["conditions"]:
                continue
            for grp in ("direct", "branch"):
                for pair, sc in pd["conditions"][arm][grp].items():
                    r[f"{arm}:{grp}:{pair}"] = round(_mean(_paired_shift(clean[grp][pair], sc)), 3)
        out["remediation"][pn] = r
    json.dump(out, open(os.path.join(RESULTS, os.environ.get("LOYALTY_SWEEP_OUT", "sweep_control.json")), "w"), indent=1)
    print("\nremediation (arm:group:pair -> mean shift vs clean):")
    for pn, r in out["remediation"].items():
        print(" ", pn, {k.split(':',1)[0][:3] + ':' + k.split(':')[1][:3] + ':' + k.split(':')[2].split(' ')[0]: v for k, v in r.items()})

if __name__ == "__main__":
    main()
