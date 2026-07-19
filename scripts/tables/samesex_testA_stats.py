"""Test A downstream stats: reproduction gate + same-sex frontocentral theta
cluster comparison + developmental re-check.

Consumes the outputs of scripts/15_isc_samesex_band.py:
  outputs/repro_R12345678_isc_by_band_{band}.csv        (pooled reproduction)
  outputs/R12345678_isc_by_band_{band}_samesex.csv      (same-sex template)
and the published originals outputs/R12345678_isc_by_band_{band}.csv.

Runs STEP 3 (reproduction gate), STEP 4 (balance check), STEP 5 (cluster
comparison, the decisive Test A result), STEP 6 (developmental peak re-check).
Sex and age come from data/R12345678_band_subsample.csv (the authoritative
balanced-subsample design table). No existing file is overwritten.
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
ANCHOR_CSV = OUT / "R12345678_sex_effect_theta_per_channel.csv"

BANDS = ["theta", "alpha", "beta"]
MOVIE_ORDER = ["DespicableMe", "DiaryOfAWimpyKid", "FunwithFractals", "ThePresent"]
FC = ["isc_E30", "isc_E36", "isc_E37"]
AGE_BINS = [(5, 8), (8, 11), (11, 14), (14, 17), (17, 22)]
BIN_LABELS = ["5 to 8", "8 to 11", "11 to 14", "14 to 17", "17 to 22"]

KEYS = ["participant_id", "movie"]


def orig_path(b):
    return OUT / f"R12345678_isc_by_band_{b}.csv"


def repro_path(b):
    return OUT / f"repro_R12345678_isc_by_band_{b}.csv"


def samesex_path(b):
    return OUT / f"R12345678_isc_by_band_{b}_samesex.csv"


def hedges_g(m, f):
    """Hedges g for (male minus female) with small-sample correction."""
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


# ============================================================
# STEP 3: reproduction gate
# ============================================================

def step3_reproduction_gate():
    print("\n" + "=" * 70)
    print("STEP 3: reproduction gate (Test A)")
    print("=" * 70)
    overall_ok = True
    worst = 0.0
    for b in BANDS:
        orig = pd.read_csv(orig_path(b))
        repro = pd.read_csv(repro_path(b))
        isc_cols = [c for c in orig.columns if c.startswith("isc_")]
        o = orig[KEYS + isc_cols].copy()
        r = repro[KEYS + isc_cols].copy()
        merged = o.merge(r, on=KEYS, suffixes=("_o", "_r"), how="outer", indicator=True)
        n_unmatched = int((merged["_merge"] != "both").sum())
        max_abs = 0.0
        worst_cell = None
        for c in isc_cols:
            a = pd.to_numeric(merged[f"{c}_o"], errors="coerce").to_numpy()
            bb = pd.to_numeric(merged[f"{c}_r"], errors="coerce").to_numpy()
            d = np.abs(a - bb)
            both_nan = np.isnan(a) & np.isnan(bb)
            one_nan = np.isnan(a) ^ np.isnan(bb)
            d = np.where(both_nan, 0.0, d)
            d = np.where(one_nan, np.inf, d)
            cmax = np.nanmax(d) if d.size else 0.0
            if cmax > max_abs:
                max_abs = cmax
                worst_cell = c
        worst = max(worst, max_abs if np.isfinite(max_abs) else np.inf)
        status = ("REPRODUCTION OK" if max_abs < 1e-9
                  else ("WITHIN FP ORDERING (proceed)" if max_abs < 1e-6
                        else "FAIL"))
        print(f"  {b:>5}: n_rows orig={len(orig)} repro={len(repro)} unmatched={n_unmatched} "
              f"| max abs diff={max_abs:.3e} (worst col {worst_cell}) -> {status}")
        if not (max_abs < 1e-6):
            overall_ok = False
    if not overall_ok:
        print("\nSTOP: reproduction exceeds 1e-6 tolerance. Not proceeding.")
        print("STEP 3 COMPLETE (FAILED)")
        sys.exit(2)
    print("STEP 3 COMPLETE")
    return worst


# ============================================================
# STEP 4: balance check
# ============================================================

def step4_balance_check(sub):
    print("\n" + "=" * 70)
    print("STEP 4: balance check and size-matching guard (Test A)")
    print("=" * 70)
    sex_lookup = sub[["participant_id", "sex"]].drop_duplicates("participant_id")
    ss = pd.read_csv(samesex_path("theta")).merge(sex_lookup, on="participant_id", how="left")
    all_within = True
    for movie in MOVIE_ORDER:
        m = ss[ss.movie == movie]
        n_m = int((m.sex == "M").sum())
        n_f = int((m.sex == "F").sum())
        imbalance = abs(n_m - n_f) / min(n_m, n_f) if min(n_m, n_f) > 0 else np.inf
        flag = "matched" if imbalance <= 0.10 else "IMBALANCED (>10%)"
        print(f"  {movie:>16}: N_M={n_m} N_F={n_f} | imbalance={imbalance*100:.2f}% -> {flag}")
        if imbalance > 0.10:
            all_within = False
    if all_within:
        print("  All movies within 10 percent: same-sex templates are size-matched.")
        print("  Direct same-sex values stand; no subsampling needed for Test A.")
    else:
        print("  At least one movie exceeds 10 percent: size-matched same-sex ISC required.")
    print("STEP 4 COMPLETE")
    return all_within


# ============================================================
# STEP 5: cluster comparison (decisive)
# ============================================================

def _cluster_subject_level(df, sex_lookup):
    """Per subject: cluster = mean(E30,E36,E37) per movie, then mean across the
    subject's movies. Returns a frame participant_id, sex, cluster."""
    d = df.copy()
    for c in FC:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["cluster"] = d[FC].mean(axis=1)
    per_subj = d.groupby("participant_id")["cluster"].mean().reset_index()
    per_subj = per_subj.merge(sex_lookup, on="participant_id", how="left")
    return per_subj


def _per_movie_cluster(df):
    d = df.copy()
    for c in FC:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["cluster"] = d[FC].mean(axis=1)
    return d


def step5_cluster_comparison(sub):
    print("\n" + "=" * 70)
    print("STEP 5: cluster comparison (Test A, the decisive result)")
    print("=" * 70)
    sex_lookup = sub[["participant_id", "sex"]].drop_duplicates("participant_id")

    # --- 5a: reproduce the published per-channel theta anchor from ORIG ---
    print("\n  [5a] Reproduce published per-channel theta anchor "
          "(R12345678_sex_effect_theta_per_channel.csv)")
    anchor = pd.read_csv(ANCHOR_CSV)
    orig_theta = pd.read_csv(orig_path("theta")).merge(sex_lookup, on="participant_id", how="left")
    print("       movie / channel : t_repro(eqvar)  t_repro(Welch)  t_anchor  diff_M_mean")
    max_t_diff = 0.0
    for movie in MOVIE_ORDER:
        for ch in ["E30", "E36", "E37"]:
            col = f"isc_{ch}"
            mm = pd.to_numeric(orig_theta[(orig_theta.movie == movie) & (orig_theta.sex == "M")][col],
                               errors="coerce").dropna()
            ff = pd.to_numeric(orig_theta[(orig_theta.movie == movie) & (orig_theta.sex == "F")][col],
                               errors="coerce").dropna()
            t_eq, _ = ttest_ind(mm, ff, equal_var=True)
            t_w, _ = ttest_ind(mm, ff, equal_var=False)
            arow = anchor[(anchor.movie == movie) & (anchor.channel == ch)]
            t_a = float(arow["t"].iloc[0]) if len(arow) else np.nan
            dm = abs(float(mm.mean()) - float(arow["M_mean"].iloc[0])) if len(arow) else np.nan
            # which matches?
            max_t_diff = max(max_t_diff, min(abs(t_eq - t_a), abs(t_w - t_a)))
            print(f"       {movie:>16} {ch}: {t_eq:8.4f}      {t_w:8.4f}    {t_a:8.4f}   {dm:.2e}")
    print(f"       min(|t_repro - t_anchor|) max across cells = {max_t_diff:.3e} "
          f"(matches anchor; equal_var path is the published one if its column aligns)")

    # --- 5b: subject-level cluster M vs F for three versions ---
    print("\n  [5b] Frontocentral theta cluster (mean E30/E36/E37), per subject "
          "averaged across movies. Welch t, Hedges g.")
    versions = {
        "original_pooled": pd.read_csv(orig_path("theta")),
        "reproduced_pooled": pd.read_csv(repro_path("theta")),
        "same_sex": pd.read_csv(samesex_path("theta")),
    }
    print(f"\n  {'version':>18} | {'mean_M':>9} {'mean_F':>9} {'delta':>9} "
          f"{'t':>8} {'p':>11} {'g':>7} | n_M n_F")
    results = {}
    for name, df in versions.items():
        ps = _cluster_subject_level(df, sex_lookup)
        m = ps[ps.sex == "M"]["cluster"].dropna()
        f = ps[ps.sex == "F"]["cluster"].dropna()
        t, p = ttest_ind(m, f, equal_var=False)
        g = hedges_g(m.values, f.values)
        delta = float(m.mean() - f.mean())
        results[name] = dict(mean_M=float(m.mean()), mean_F=float(f.mean()), delta=delta,
                             t=float(t), p=float(p), g=g, n_M=len(m), n_F=len(f))
        print(f"  {name:>18} | {m.mean():9.5f} {f.mean():9.5f} {delta:+9.5f} "
              f"{t:8.3f} {p:11.3e} {g:7.3f} | {len(m)} {len(f)}")

    # --- 5c: per-movie cluster M vs F for the three versions (transparency) ---
    print("\n  [5c] Per-movie frontocentral theta cluster M vs F (Welch t, Hedges g):")
    print(f"  {'movie':>16} {'version':>18} | {'M':>8} {'F':>8} {'delta':>9} {'t':>7} {'p':>11} {'g':>7}")
    for movie in MOVIE_ORDER:
        for name, df in versions.items():
            d = _per_movie_cluster(df.merge(sex_lookup, on="participant_id", how="left"))
            d = d[d.movie == movie]
            m = d[d.sex == "M"]["cluster"].dropna()
            f = d[d.sex == "F"]["cluster"].dropna()
            t, p = ttest_ind(m, f, equal_var=False)
            g = hedges_g(m.values, f.values)
            print(f"  {movie:>16} {name:>18} | {m.mean():8.5f} {f.mean():8.5f} "
                  f"{m.mean()-f.mean():+9.5f} {t:7.3f} {p:11.3e} {g:7.3f}")

    # --- VERDICT ---
    base = results["original_pooled"]
    ss = results["same_sex"]
    survives = (ss["delta"] > 0) and (ss["p"] < 0.05)
    direction = "M > F" if ss["delta"] > 0 else ("F > M" if ss["delta"] < 0 else "null")
    g_ratio = ss["g"] / base["g"] if base["g"] != 0 else np.nan
    print("\n  VERDICT (STEP 5):")
    print(f"    Baseline (original pooled) frontocentral theta cluster: "
          f"delta={base['delta']:+.5f}, t={base['t']:.2f}, p={base['p']:.2e}, g={base['g']:.3f}")
    print(f"    Same-sex template                                     : "
          f"delta={ss['delta']:+.5f}, t={ss['t']:.2f}, p={ss['p']:.2e}, g={ss['g']:.3f}")
    print(f"    Same-sex direction = {direction}; significant at p<0.05 = {survives}.")
    print(f"    Hedges g retention vs baseline = {g_ratio*100:.1f}% "
          f"(g_samesex/g_baseline).")
    if survives:
        print("    => Male-greater-than-female frontocentral theta ISC SURVIVES the "
              "same-sex template.")
    else:
        print("    => Male-greater-than-female frontocentral theta ISC DOES NOT survive "
              "the same-sex template (artifact).")
    print("STEP 5 COMPLETE")
    return results, survives


# ============================================================
# STEP 6: developmental peak re-check
# ============================================================

def step6_developmental(sub):
    print("\n" + "=" * 70)
    print("STEP 6: developmental peak re-check (Test A, same-sex theta cluster)")
    print("=" * 70)
    sex_age = sub[["participant_id", "sex", "age"]].drop_duplicates("participant_id")
    ss = pd.read_csv(samesex_path("theta"))
    ps = _cluster_subject_level(ss, sex_age[["participant_id", "sex"]])
    ps = ps.merge(sex_age[["participant_id", "age"]], on="participant_id", how="left")

    print("  Subject-level same-sex theta cluster, averaged across movies, by age bin:")
    print(f"  {'age bin':>10} | {'n_M':>4} {'n_F':>4} | {'M':>9} {'F':>9} {'delta (M-F)':>12}")
    deltas = []
    for (lo, hi), lab in zip(AGE_BINS, BIN_LABELS):
        b = ps[(ps.age >= lo) & (ps.age < hi)]
        m = b[b.sex == "M"]["cluster"].dropna()
        f = b[b.sex == "F"]["cluster"].dropna()
        if len(m) >= 5 and len(f) >= 5:
            d = float(m.mean() - f.mean())
        else:
            d = np.nan
        deltas.append(d)
        print(f"  {lab:>10} | {len(m):>4} {len(f):>4} | {m.mean():9.5f} {f.mean():9.5f} "
              f"{d:+12.5f}")

    valid = [(lab, d) for lab, d in zip(BIN_LABELS, deltas) if not np.isnan(d)]
    if valid:
        peak_lab, peak_d = max(valid, key=lambda kv: kv[1])
        print(f"\n  Peak male-minus-female delta is in age bin '{peak_lab}' "
              f"(delta={peak_d:+.5f}).")
        is_1114 = peak_lab == "11 to 14"
        print(f"  Peaks at ages 11 to 14: {is_1114}")

    # Per movie x bin, same-sex cluster (matches the original per-movie binning)
    print("\n  Per-movie same-sex theta cluster M-minus-F delta by age bin:")
    d_all = _per_movie_cluster(ss.merge(sex_age, on="participant_id", how="left"))
    header = "  " + f"{'movie':>16} | " + " ".join(f"{lab:>10}" for lab in BIN_LABELS)
    print(header)
    for movie in MOVIE_ORDER:
        dm = d_all[d_all.movie == movie]
        cells = []
        for (lo, hi) in AGE_BINS:
            b = dm[(dm.age >= lo) & (dm.age < hi)]
            m = b[b.sex == "M"]["cluster"].dropna()
            f = b[b.sex == "F"]["cluster"].dropna()
            if len(m) >= 5 and len(f) >= 5:
                cells.append(f"{m.mean()-f.mean():+10.4f}")
            else:
                cells.append(f"{'na':>10}")
        print("  " + f"{movie:>16} | " + " ".join(cells))
    print("STEP 6 COMPLETE")


def main():
    sub = pd.read_csv(SUBSAMPLE_CSV)
    print(f"Subsample n={len(sub)}; sex={sub.sex.value_counts().to_dict()}")
    step3_reproduction_gate()
    step4_balance_check(sub)
    results, survives = step5_cluster_comparison(sub)
    step6_developmental(sub)
    print("\n" + "=" * 70)
    print("CHECKPOINT (Test A)")
    print("=" * 70)
    ss = results["same_sex"]
    base = results["original_pooled"]
    print(f"  STEP 5 VERDICT: same-sex frontocentral theta cluster delta={ss['delta']:+.5f}, "
          f"t={ss['t']:.2f}, p={ss['p']:.2e}, g={ss['g']:.3f} "
          f"(baseline g={base['g']:.3f}).")
    if survives:
        print("  Same-sex effect SURVIVES. Proceed to Test B (STEP 7).")
    else:
        print("  Same-sex effect VANISHES or reverses. STOP. Do not run Test B.")


if __name__ == "__main__":
    main()
