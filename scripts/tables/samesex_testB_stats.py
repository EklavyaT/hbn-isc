"""Test B downstream stats: broadband full-cohort same-sex size-matched
male-vs-female comparison, pooled vs size-matched.

Consumes the outputs of scripts/16_isc_samesex_broadband.py:
  outputs/repro_R12345678_isc_per_subject.csv                 (pooled reproduction)
  outputs/R12345678_isc_per_subject_samesex_sizematched.csv   (Test B)
and the published originals
  outputs/R12345678_isc_per_subject.csv
  outputs/R12345678_sex_effect_per_channel.csv         (broadband per-channel anchor)
  outputs/R12345678_sex_effect_extended_stats.csv      (published Welch t on grand mean)

The paper reports a broadband full-cohort male-vs-female comparison: Table 4
(per-movie M-vs-F on grand-mean ISC) and the broadband per-channel sex-effect
topography with the frontocentral cluster (E30, E36, E37). This script runs that
exact comparison for the pooled and the same-sex size-matched ISC, on both the
whole-head grand mean and the frontocentral cluster, with Welch t and Hedges g.
No existing file is overwritten.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT = PROJECT_ROOT / "outputs"
COHORT_CSV = OUT / "R12345678_isc_cohort.csv"
ORIG = OUT / "R12345678_isc_per_subject.csv"
REPRO = OUT / "repro_R12345678_isc_per_subject.csv"
SS = OUT / "R12345678_isc_per_subject_samesex_sizematched.csv"
ANCHOR = OUT / "R12345678_sex_effect_per_channel.csv"
EXT_STATS = OUT / "R12345678_sex_effect_extended_stats.csv"

MOVIE_ORDER = ["DespicableMe", "DiaryOfAWimpyKid", "FunwithFractals", "ThePresent"]
FC = ["isc_E30", "isc_E36", "isc_E37"]


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


def add_cluster(df):
    d = df.copy()
    for c in FC:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["cluster"] = d[FC].mean(axis=1)
    d["grand"] = pd.to_numeric(d["isc_grand_mean"], errors="coerce")
    return d


def welch_table(d, value_col, label):
    """Per-movie and across-movie subject-level Welch t + Hedges g on value_col."""
    print(f"\n  [{label}] per movie:")
    print(f"  {'movie':>16} | {'M':>9} {'F':>9} {'delta':>10} {'t':>8} {'p':>11} {'g':>7} | nM nF")
    for movie in MOVIE_ORDER:
        sub = d[d.movie == movie]
        m = sub[sub.sex == "M"][value_col].dropna()
        f = sub[sub.sex == "F"][value_col].dropna()
        t, p = ttest_ind(m, f, equal_var=False)
        g = hedges_g(m.values, f.values)
        print(f"  {movie:>16} | {m.mean():9.5f} {f.mean():9.5f} {m.mean()-f.mean():+10.5f} "
              f"{t:8.3f} {p:11.3e} {g:7.3f} | {len(m)} {len(f)}")
    # across-movie subject level
    ps = d.groupby(["participant_id", "sex"])[value_col].mean().reset_index()
    m = ps[ps.sex == "M"][value_col].dropna()
    f = ps[ps.sex == "F"][value_col].dropna()
    t, p = ttest_ind(m, f, equal_var=False)
    g = hedges_g(m.values, f.values)
    res = dict(mean_M=float(m.mean()), mean_F=float(f.mean()),
               delta=float(m.mean() - f.mean()), t=float(t), p=float(p), g=g,
               n_M=len(m), n_F=len(f))
    print(f"  across-movie subject mean: M={res['mean_M']:.5f} F={res['mean_F']:.5f} "
          f"delta={res['delta']:+.5f} t={res['t']:.3f} p={res['p']:.3e} g={res['g']:.3f}")
    return res


def main():
    cohort = pd.read_csv(COHORT_CSV)
    sex_lookup = cohort[["participant_id", "sex"]].drop_duplicates("participant_id")

    print("=" * 70)
    print("STEP 7 stats: broadband full-cohort same-sex size-matched (Test B)")
    print("=" * 70)

    # --- reproduction sanity vs original (gate already enforced in script 16) ---
    orig = pd.read_csv(ORIG)
    repro = pd.read_csv(REPRO)
    isc_cols = [c for c in orig.columns if c.startswith("isc_")]
    mg = orig[["participant_id", "movie"] + isc_cols].merge(
        repro[["participant_id", "movie"] + isc_cols], on=["participant_id", "movie"],
        suffixes=("_o", "_r"), how="outer", indicator=True)
    n_unmatched = int((mg["_merge"] != "both").sum())
    max_abs = 0.0
    for c in isc_cols:
        a = pd.to_numeric(mg[f"{c}_o"], errors="coerce").to_numpy()
        b = pd.to_numeric(mg[f"{c}_r"], errors="coerce").to_numpy()
        dd = np.abs(a - b)
        both = np.isnan(a) & np.isnan(b)
        one = np.isnan(a) ^ np.isnan(b)
        dd = np.where(both, 0.0, dd)
        dd = np.where(one, np.inf, dd)
        max_abs = max(max_abs, np.nanmax(dd) if dd.size else 0.0)
    print(f"\n  Reproduction (repro vs original per-subject): unmatched={n_unmatched} "
          f"max abs diff={max_abs:.3e} -> "
          f"{'OK' if max_abs < 1e-9 else ('WITHIN FP' if max_abs < 1e-6 else 'FAIL')}")

    # --- reproduce broadband per-channel sex anchor at frontocentral cluster ---
    print("\n  Reproduce published broadband sex anchor at frontocentral cluster "
          "(R12345678_sex_effect_per_channel.csv):")
    anchor = pd.read_csv(ANCHOR)
    od = orig.merge(sex_lookup, on="participant_id", how="left")
    print(f"  {'movie':>16} {'ch':>4} | {'t_repro':>9} {'t_anchor':>9} {'dM':>9}")
    for movie in MOVIE_ORDER:
        for ch in ["E30", "E36", "E37"]:
            col = f"isc_{ch}"
            m = pd.to_numeric(od[(od.movie == movie) & (od.sex == "M")][col], errors="coerce").dropna()
            f = pd.to_numeric(od[(od.movie == movie) & (od.sex == "F")][col], errors="coerce").dropna()
            t, _ = ttest_ind(m, f, equal_var=True)
            ar = anchor[(anchor.movie == movie) & (anchor.channel == ch)]
            ta = float(ar["t"].iloc[0]) if len(ar) else np.nan
            dM = abs(float(m.mean()) - float(ar["M_mean"].iloc[0])) if len(ar) else np.nan
            print(f"  {movie:>16} {ch:>4} | {t:9.4f} {ta:9.4f} {dM:9.2e}")

    # --- published grand-mean Welch t (for context) ---
    if EXT_STATS.exists():
        ext = pd.read_csv(EXT_STATS)
        print("\n  Published grand-mean Welch t per movie (R12345678_sex_effect_extended_stats.csv):")
        for _, r in ext.iterrows():
            print(f"    {r['movie']:>16}: welch_t={r['welch_t']:.3f} d={r['cohens_d']:+.3f} "
                  f"nM={int(r['n_M'])} nF={int(r['n_F'])}")

    # --- versions ---
    versions = {
        "original_pooled": add_cluster(orig.merge(sex_lookup, on="participant_id", how="left")),
        "reproduced_pooled": add_cluster(repro.merge(sex_lookup, on="participant_id", how="left")),
        "same_sex_sizematched": add_cluster(pd.read_csv(SS)),  # already has sex col
    }

    summary = {}
    for metric, vcol, mlabel in [("grand", "grand", "whole-head grand-mean ISC"),
                                 ("cluster", "cluster", "frontocentral cluster ISC")]:
        print("\n" + "-" * 70)
        print(f"  METRIC: {mlabel}")
        print("-" * 70)
        for name, d in versions.items():
            res = welch_table(d, vcol, f"{name}")
            summary[(metric, name)] = res

    # --- VERDICT ---
    print("\n" + "=" * 70)
    print("  VERDICT (STEP 7, Test B)")
    print("=" * 70)
    for metric, mlabel in [("grand", "whole-head grand mean"),
                           ("cluster", "frontocentral cluster")]:
        base = summary[(metric, "original_pooled")]
        ss = summary[(metric, "same_sex_sizematched")]
        survives = (ss["delta"] > 0) and (ss["p"] < 0.05)
        ratio = ss["g"] / base["g"] if base["g"] != 0 else np.nan
        direction = "M > F" if ss["delta"] > 0 else ("F > M" if ss["delta"] < 0 else "null")
        print(f"\n  {mlabel}:")
        print(f"    pooled (2:1 skew)        : delta={base['delta']:+.5f}, t={base['t']:.2f}, "
              f"p={base['p']:.2e}, g={base['g']:.3f}")
        print(f"    same-sex size-matched    : delta={ss['delta']:+.5f}, t={ss['t']:.2f}, "
              f"p={ss['p']:.2e}, g={ss['g']:.3f}")
        print(f"    direction={direction}, significant={survives}, "
              f"g retention={ratio*100:.1f}% of pooled.")
    print("\nSTEP 7 COMPLETE")


if __name__ == "__main__":
    main()
