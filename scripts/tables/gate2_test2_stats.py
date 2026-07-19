"""Gate 2 (ocular artifact) Tier 1, Test 2 statistics: delta-band discriminator.

Logic: blink and saccade residual carries most power in delta (1 to 4 Hz). If the
male greater than female sex effect is larger in delta than in theta, that is the
blink-spectrum signature. If delta is weaker than theta, the effect is genuinely
theta-centered (frontal midline theta), not a low-frequency ocular tail.

Consumes the delta band ISC from scripts/17_isc_delta_band.py
(outputs/R12345678_isc_by_band_delta.csv) and the published band CSVs
(outputs/R12345678_isc_by_band_{theta,alpha,beta}.csv), with sex from
data/R12345678_band_subsample.csv. Theta is recomputed here as a within-run
anchor (its frontocentral cluster Hedges g should match the gate 1 baseline of
0.889). Writes outputs/gate2_test2_delta_vs_theta.csv. Overwrites nothing else.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT = PROJECT_ROOT / "outputs"
SUBSAMPLE_CSV = PROJECT_ROOT / "data" / "R12345678_band_subsample.csv"
RESULT_CSV = OUT / "gate2_test2_delta_vs_theta.csv"

# low to high; delta is new, theta is the published peak, alpha and beta the
# published high-side nulls (printed for the full profile, not in the test CSV).
TEST_BANDS = ["delta", "theta"]
PROFILE_BANDS = ["delta", "theta", "alpha", "beta"]
MOVIE_ORDER = ["DespicableMe", "DiaryOfAWimpyKid", "FunwithFractals", "ThePresent"]
FC = ["isc_E30", "isc_E36", "isc_E37"]
GATE1_THETA_G = 0.889  # gate 1 baseline frontocentral theta cluster Hedges g


def band_path(b):
    return OUT / f"R12345678_isc_by_band_{b}.csv"


def hedges_g(m, f):
    m = np.asarray(m, dtype=float)
    f = np.asarray(f, dtype=float)
    nm, nf = len(m), len(f)
    vm, vf = m.var(ddof=1), f.var(ddof=1)
    sp = np.sqrt(((nm - 1) * vm + (nf - 1) * vf) / (nm + nf - 2))
    if sp == 0:
        return 0.0
    d = (m.mean() - f.mean()) / sp
    J = 1.0 - 3.0 / (4.0 * (nm + nf) - 9.0)
    return d * J


def subject_level(df, sex_lookup, value):
    """Per subject value averaged across movies, with sex. value is 'cluster'
    (mean of E30/E36/E37) or 'grand' (whole-head isc_grand_mean)."""
    d = df.copy()
    if value == "cluster":
        for c in FC:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d["v"] = d[FC].mean(axis=1)
    else:
        d["v"] = pd.to_numeric(d["isc_grand_mean"], errors="coerce")
    per = d.groupby("participant_id")["v"].mean().reset_index()
    return per.merge(sex_lookup, on="participant_id", how="left")


def mvf(per):
    m = per[per.sex == "M"]["v"].dropna()
    f = per[per.sex == "F"]["v"].dropna()
    t, p = ttest_ind(m, f, equal_var=False)
    return dict(mean_M=float(m.mean()), mean_F=float(f.mean()),
               delta=float(m.mean() - f.mean()), t=float(t), p=float(p),
               g=hedges_g(m.values, f.values), n_M=len(m), n_F=len(f))


def main():
    if not band_path("delta").exists():
        print(f"STOP: delta band file not found: {band_path('delta')}")
        print("Run scripts/17_isc_delta_band.py first.")
        sys.exit(1)
    sub = pd.read_csv(SUBSAMPLE_CSV)
    sex_lookup = sub[["participant_id", "sex"]].drop_duplicates("participant_id")
    print(f"Subsample n={len(sub)}; sex={sub.sex.value_counts().to_dict()}")

    print("\n" + "=" * 70)
    print("GATE 2 TIER 1, STEP 3: delta vs theta frontocentral cluster and grand mean")
    print("=" * 70)

    # within-run theta anchor: cluster g should match gate 1 baseline 0.889
    theta_cluster = mvf(subject_level(pd.read_csv(band_path("theta")), sex_lookup, "cluster"))
    print(f"\n  within-run theta anchor: frontocentral cluster Hedges g="
          f"{theta_cluster['g']:.3f} (gate 1 baseline {GATE1_THETA_G}); "
          f"match={abs(theta_cluster['g'] - GATE1_THETA_G) < 0.01}")

    rows = []
    print(f"\n  {'band':>6} {'metric':>14} | {'mean_M':>9} {'mean_F':>9} {'delta':>10} "
          f"{'Welch t':>9} {'p':>11} {'g':>8} | n_M n_F")
    for b in TEST_BANDS:
        for metric, label in [("cluster", "frontocentral"), ("grand", "whole-head grand")]:
            r = mvf(subject_level(pd.read_csv(band_path(b)), sex_lookup, metric))
            rows.append({"band": b, "metric": metric, **r})
            print(f"  {b:>6} {label:>14} | {r['mean_M']:9.5f} {r['mean_F']:9.5f} "
                  f"{r['delta']:+10.5f} {r['t']:9.3f} {r['p']:11.3e} {r['g']:8.3f} "
                  f"| {r['n_M']} {r['n_F']}")

    # full spectral profile at the cluster (context for the memo)
    print(f"\n  full spectral profile, frontocentral cluster Hedges g:")
    prof = {}
    for b in PROFILE_BANDS:
        if band_path(b).exists():
            prof[b] = mvf(subject_level(pd.read_csv(band_path(b)), sex_lookup, "cluster"))["g"]
            print(f"    {b:>6}: g={prof[b]:+.3f}")

    # verdict
    dg = next(r for r in rows if r["band"] == "delta" and r["metric"] == "cluster")["g"]
    tg = next(r for r in rows if r["band"] == "theta" and r["metric"] == "cluster")["g"]
    print("\n  VERDICT (Test 2):")
    print(f"    frontocentral cluster Hedges g: theta={tg:.3f}, delta={dg:.3f} "
          f"(delta is {dg/tg*100:.1f} percent of theta).")
    blink_consistent = dg >= tg
    print(f"    blink prediction (delta g >= theta g) holds: {blink_consistent}")
    if not blink_consistent:
        print("    => The sex effect is theta-peaked, not a low-frequency blink tail.")
        print("       Delta is weaker than theta, so the spectral profile does not")
        print("       match the blink spectrum. Argues against an ocular generator.")
    else:
        print("    => Delta effect is comparable to or larger than theta. BLINK-")
        print("       CONSISTENT spectral profile. Flagged, not softened.")

    cols = ["band", "metric", "mean_M", "mean_F", "delta", "t", "p", "g", "n_M", "n_F"]
    pd.DataFrame(rows)[cols].to_csv(RESULT_CSV, index=False)
    print(f"\n  wrote {RESULT_CSV}")
    print("STEP 3 COMPLETE")


if __name__ == "__main__":
    main()
