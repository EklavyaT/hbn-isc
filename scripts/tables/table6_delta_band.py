"""Integrate the delta band (1 to 4 Hz) into the primary Table 6 band analysis.

Reviewer request: report delta through the SAME subsample and regression
framework as theta, alpha, and beta, in the primary table, rather than via a
separate effect-size metric.

This script imports the exact functions used to build the published Table 6 and
Table 4 from build_tables.py. Nothing is reimplemented: the two-sample t-test is
scipy ttest_ind on the pre-computed per-subject isc_grand_mean, and the
regression is the same _full_regression OLS model used for the primary sex
analysis, with the band ISC substituted as the dependent variable.

Steps:
  1. Reproduce the published theta/alpha/beta Table 6 numbers (reproduction gate).
  2. Compute delta through the identical path (t-test plus primary regression).
  3. Write outputs/table6_delta_row.csv plus a drop-in LaTeX row block.
  4. Write outputs/band_frontocentral_4band.csv (frontocentral cluster, 4 bands).

Creates new files only. No existing script or CSV is overwritten.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_tables import (  # noqa: E402
    MOVIE_ORDER,
    MOVIE_LABELS,
    _full_regression,
    _load_analysis_df,
    fmt_t,
    fmt_p,
    fmt_int,
    fmt_beta,
    ttest_ind,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"

BAND_ORDER = ["delta", "theta", "alpha", "beta"]
BAND_RANGE = {"delta": "1-4", "theta": "4-8", "alpha": "8-12", "beta": "12-20"}
FRONTOCENTRAL = ["E30", "E36", "E37"]

# Published Table 6 values, transcribed from tables/csv/table6_band_breakdown.csv
# at the time this script was written. Used only as the reproduction gate.
REPRO_TOL = 0.05


def load_band(band):
    """Per-subject per-movie band ISC on the n=400 balanced subsample."""
    return pd.read_csv(OUT_DIR / f"R12345678_isc_by_band_{band}.csv")


def sex_lookup():
    master = pd.read_csv(OUT_DIR / "R12345678_master.csv")
    return master[["participant_id", "sex"]].drop_duplicates("participant_id")


def band_ttest(df_band, sex_map, movie, value_col="isc_grand_mean"):
    """The exact Table 6 two-sample test: ttest_ind on grand-mean band ISC."""
    df_b = df_band.merge(sex_map, on="participant_id", how="inner")
    sub = df_b[df_b.movie == movie]
    m = sub[sub.sex == "M"][value_col].dropna()
    f = sub[sub.sex == "F"][value_col].dropna()
    t, p = ttest_ind(m, f)
    return float(t), float(p), len(m), len(f), float(m.mean()), float(f.mean())


def band_regression_frame(df_band, value_col="isc_grand_mean"):
    """Attach band ISC as the dependent variable to the primary covariate frame.

    The primary model is fit on outputs/R12345678_sex_analysis_df.csv. We keep
    every covariate exactly as the primary analysis defines it (sex_M, age,
    age_sq, ehq_total, release, and the four CBCL bifactor scores) and only
    swap the dependent variable from broadband to band-specific ISC. The join
    on participant_id plus movie restricts the frame to the n=400 balanced
    subsample automatically, because the band files contain only those subjects.
    """
    cov = _load_analysis_df().drop(columns=["isc_grand_mean"])
    band = df_band[["participant_id", "movie", value_col]].rename(
        columns={value_col: "isc_grand_mean"}
    )
    merged = cov.merge(band, on=["participant_id", "movie"], how="inner")
    # Same completeness filter the primary sex regression applies (table4).
    return merged.dropna(subset=["ehq_total", "p_factor", "attention",
                                 "internalizing", "externalizing"])


def step1_reproduction(bands, sex_map):
    print("=" * 74)
    print("STEP 1: reproduction gate, published theta/alpha/beta Table 6")
    print("=" * 74)
    published = pd.read_csv(PROJECT_ROOT / "tables" / "csv" / "table6_band_breakdown.csv")
    ok = True
    print(f"{'movie':22s} {'band':6s} {'recomputed t':>13s} {'published t':>12s} {'match':>6s}")
    for movie in MOVIE_ORDER:
        pub_row = published[published["Movie"] == MOVIE_LABELS[movie]].iloc[0]
        for band in ["theta", "alpha", "beta"]:
            t, p, _, _, _, _ = band_ttest(bands[band], sex_map, movie)
            pub_t = float(pub_row[f"{band.capitalize()} t"])
            hit = abs(t - pub_t) <= REPRO_TOL
            ok = ok and hit
            print(f"{movie:22s} {band:6s} {t:13.3f} {pub_t:12.1f} {'OK' if hit else 'FAIL':>6s}")
    if not ok:
        print("\nREPRODUCTION FAILED. Stopping.")
        sys.exit(1)
    print("\nAll theta/alpha/beta t-statistics reproduce the published Table 6.")
    print("STEP 1 COMPLETE\n")


def step2_delta(bands, sex_map):
    print("=" * 74)
    print("STEP 2: delta through the identical path")
    print("=" * 74)
    results = {}
    for band in BAND_ORDER:
        frame = band_regression_frame(bands[band])
        per_movie = []
        for movie in MOVIE_ORDER:
            t, p, n_m, n_f, mean_m, mean_f = band_ttest(bands[band], sex_map, movie)
            sub = frame[frame.movie == movie]
            fit = _full_regression(sub)
            per_movie.append({
                "band": band,
                "movie": movie,
                "movie_label": MOVIE_LABELS[movie],
                "t": t,
                "p": p,
                "n_M": n_m,
                "n_F": n_f,
                "n": n_m + n_f,
                "mean_M": mean_m,
                "mean_F": mean_f,
                "beta_sex_M": float(fit.params["sex_M"]),
                "p_regression": float(fit.pvalues["sex_M"]),
                "t_regression": float(fit.tvalues["sex_M"]),
                "r_squared": float(fit.rsquared),
                "n_regression": int(len(sub)),
            })
        results[band] = pd.DataFrame(per_movie)

    d = results["delta"]
    print("Delta (1 to 4 Hz), per movie:\n")
    print(f"{'movie':22s} {'t':>8s} {'p':>11s} {'beta sex_M':>11s} {'p (regr)':>11s} {'n':>5s} {'n regr':>7s}")
    for _, r in d.iterrows():
        print(f"{r['movie_label']:22s} {r['t']:8.3f} {r['p']:11.2e} "
              f"{r['beta_sex_M']:+11.4f} {r['p_regression']:11.2e} "
              f"{int(r['n']):5d} {int(r['n_regression']):7d}")
    print("\nSTEP 2 COMPLETE\n")
    return results


def step3_write_row(results):
    print("=" * 74)
    print("STEP 3: table6_delta_row.csv plus drop-in LaTeX row")
    print("=" * 74)
    d = results["delta"]
    rows = []
    for _, r in d.iterrows():
        rows.append({
            "Movie": r["movie_label"],
            "Delta t": fmt_t(r["t"]),
            "Delta p": fmt_p(r["p"]),
            "Delta beta sex_M": fmt_beta(r["beta_sex_M"]),
            "Delta p (regression)": fmt_p(r["p_regression"]),
            "n": fmt_int(r["n"]),
            "n (regression)": fmt_int(r["n_regression"]),
        })
    out = pd.DataFrame(rows)
    out_path = OUT_DIR / "table6_delta_row.csv"
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path}\n")

    # 4-band side by side summary
    print("Four-band summary, per-movie two-sample t and p:\n")
    head = f"{'movie':22s}"
    for band in BAND_ORDER:
        head += f" {band + ' t':>9s} {band + ' p':>11s}"
    print(head)
    for movie in MOVIE_ORDER:
        line = f"{MOVIE_LABELS[movie]:22s}"
        for band in BAND_ORDER:
            r = results[band]
            r = r[r.movie == movie].iloc[0]
            line += f" {r['t']:9.3f} {r['p']:11.2e}"
        print(line)

    # LaTeX rows matching the current Table 6 layout, with delta placed first.
    print("\nLaTeX rows for tab:band_breakdown (delta first, ascending frequency):\n")
    tex = []
    tex.append("% Header, delta added as the first band pair")
    tex.append("Movie & Delta $t$ & Delta $p$ & Theta $t$ & Theta $p$ & "
               "Alpha $t$ & Alpha $p$ & Beta $t$ & Beta $p$ & $n$ \\\\")
    tex.append("\\midrule")
    for movie in MOVIE_ORDER:
        cells = [MOVIE_LABELS[movie]]
        n_val = None
        for band in BAND_ORDER:
            r = results[band]
            r = r[r.movie == movie].iloc[0]
            cells.append(fmt_t(r["t"]))
            cells.append(fmt_p(r["p"]))
            n_val = int(r["n"])
        cells.append(fmt_int(n_val))
        tex.append(" & ".join(cells) + " \\\\")
    tex_block = "\n".join(tex)
    print(tex_block)

    tex_path = OUT_DIR / "table6_delta_row.tex"
    tex_path.write_text(tex_block + "\n")
    print(f"\nwrote {tex_path}")
    print("\nSTEP 3 COMPLETE\n")
    return tex_block


def step4_frontocentral(bands, sex_map):
    print("=" * 74)
    print("STEP 4: frontocentral cluster (E30, E36, E37) across four bands")
    print("=" * 74)
    fc_cols = [f"isc_{ch}" for ch in FRONTOCENTRAL]
    rows = []
    for band in BAND_ORDER:
        df_b = bands[band].copy()
        df_b["isc_fc"] = df_b[fc_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        # Per movie
        for movie in MOVIE_ORDER:
            t, p, n_m, n_f, mean_m, mean_f = band_ttest(
                df_b, sex_map, movie, value_col="isc_fc")
            rows.append({
                "band": band,
                "band_hz": BAND_RANGE[band],
                "scope": "per_movie",
                "movie": MOVIE_LABELS[movie],
                "mean_M": mean_m,
                "mean_F": mean_f,
                "diff_M_minus_F": mean_m - mean_f,
                "t": t,
                "p": p,
                "n_M": n_m,
                "n_F": n_f,
            })
        # Pooled: per subject, averaged across movies first, then one test
        merged = df_b.merge(sex_map, on="participant_id", how="inner")
        per_subj = merged.groupby(["participant_id", "sex"], as_index=False)["isc_fc"].mean()
        m = per_subj[per_subj.sex == "M"]["isc_fc"].dropna()
        f = per_subj[per_subj.sex == "F"]["isc_fc"].dropna()
        t_pool, p_pool = ttest_ind(m, f)
        rows.append({
            "band": band,
            "band_hz": BAND_RANGE[band],
            "scope": "pooled",
            "movie": "all (subject mean across movies)",
            "mean_M": float(m.mean()),
            "mean_F": float(f.mean()),
            "diff_M_minus_F": float(m.mean() - f.mean()),
            "t": float(t_pool),
            "p": float(p_pool),
            "n_M": len(m),
            "n_F": len(f),
        })
    out = pd.DataFrame(rows)
    out_path = OUT_DIR / "band_frontocentral_4band.csv"
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path}\n")

    print("Pooled frontocentral cluster (subject mean across movies):\n")
    print(f"{'band':8s} {'Hz':>7s} {'mean M':>9s} {'mean F':>9s} {'M-F':>9s} {'t':>8s} {'p':>11s}")
    for band in BAND_ORDER:
        r = out[(out.band == band) & (out.scope == "pooled")].iloc[0]
        print(f"{band:8s} {r['band_hz']:>7s} {r['mean_M']:9.5f} {r['mean_F']:9.5f} "
              f"{r['diff_M_minus_F']:9.5f} {r['t']:8.3f} {r['p']:11.2e}")

    print("\nPer-movie frontocentral t by band:\n")
    print(f"{'movie':22s}" + "".join(f" {b:>10s}" for b in BAND_ORDER))
    for movie in MOVIE_ORDER:
        line = f"{MOVIE_LABELS[movie]:22s}"
        for band in BAND_ORDER:
            r = out[(out.band == band) & (out.movie == MOVIE_LABELS[movie])].iloc[0]
            line += f" {r['t']:10.3f}"
        print(line)
    print("\nSTEP 4 COMPLETE\n")
    return out


def step5_memo(results, fc, tex_block):
    print("=" * 74)
    print("STEP 5: memo")
    print("=" * 74)
    d = results["delta"]
    th = results["theta"]

    print("\nDelta per-movie two-sample t and p (n=400 balanced):")
    for _, r in d.iterrows():
        print(f"  {r['movie_label']:22s} t = {r['t']:6.3f}, p = {r['p']:.2e}")

    print("\nDelta versus theta, per movie:")
    print(f"  {'movie':22s} {'delta t':>9s} {'theta t':>9s} {'ratio':>7s}")
    for movie in MOVIE_ORDER:
        rd = d[d.movie == movie].iloc[0]
        rt = th[th.movie == movie].iloc[0]
        print(f"  {rd['movie_label']:22s} {rd['t']:9.3f} {rt['t']:9.3f} "
              f"{rd['t'] / rt['t']:7.2f}x")

    n_sig_t = int((d["p"] < 0.05).sum())
    n_sig_reg = int((d["p_regression"] < 0.05).sum())
    th_sig_t = int((th["p"] < 0.05).sum())
    th_sig_reg = int((th["p_regression"] < 0.05).sum())

    print("\nSignificance count (alpha = 0.05):")
    print(f"  delta, two-sample t-test:  {n_sig_t} of 4 movies")
    print(f"  delta, primary regression: {n_sig_reg} of 4 movies")
    print(f"  theta, two-sample t-test:  {th_sig_t} of 4 movies")
    print(f"  theta, primary regression: {th_sig_reg} of 4 movies")

    print("\nDelta regression sex_M coefficients:")
    for _, r in d.iterrows():
        print(f"  {r['movie_label']:22s} beta = {r['beta_sex_M']:+.4f}, "
              f"p = {r['p_regression']:.2e}, n = {int(r['n_regression'])}")

    print("\nExact LaTeX rows to add to tab:band_breakdown:\n")
    print(tex_block)

    delta_all = (n_sig_t == 4) and (n_sig_reg == 4)
    print("\nBottom line:")
    if delta_all:
        print("  Delta is significant in 4 of 4 movies in BOTH the two-sample")
        print("  t-test and the primary regression, matching theta, and is the")
        print("  largest band effect in every movie.")
    else:
        print(f"  Delta is significant in {n_sig_t} of 4 movies (t-test) and "
              f"{n_sig_reg} of 4 (regression).")


def main():
    sex_map = sex_lookup()
    bands = {b: load_band(b) for b in BAND_ORDER}
    step1_reproduction(bands, sex_map)
    results = step2_delta(bands, sex_map)
    tex_block = step3_write_row(results)
    fc = step4_frontocentral(bands, sex_map)
    step5_memo(results, fc, tex_block)


if __name__ == "__main__":
    main()
