"""Build the per-channel sex-effect tables (broadband and theta).

Reconstruction of two files that existed on disk with no generating script:
  outputs/R12345678_sex_effect_per_channel.csv        (broadband, Figure 4B, Table 5)
  outputs/R12345678_sex_effect_theta_per_channel.csv  (theta, Figure 4D, gate anchor)

Both are per-movie per-channel two-sample sex contrasts on ISC:
  - Student two-sample t (scipy ttest_ind, equal_var=True), males vs females.
  - Benjamini-Hochberg FDR applied WITHIN each movie across the 129 channels,
    not globally across all 516 rows. This was determined empirically: the
    within-movie scope reproduces the stored q values to 9e-17, the global scope
    is off by 5e-3.

Broadband draws on the full ISC cohort (n_M=765, n_F=378) from
outputs/R12345678_isc_per_subject.csv. Theta draws on the balanced 400 subject
subsample (n_M=200, n_F=200) from outputs/R12345678_isc_by_band_theta.csv.

Usage:
    python scripts/build_sex_effect_per_channel.py --verify
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"
MASTER = OUT_DIR / "R12345678_master.csv"

MOVIE_ORDER = ["DespicableMe", "DiaryOfAWimpyKid", "FunwithFractals", "ThePresent"]

SPECS = {
    "broadband": {
        "source": OUT_DIR / "R12345678_isc_per_subject.csv",
        "original": OUT_DIR / "R12345678_sex_effect_per_channel.csv",
        "regen": OUT_DIR / "R12345678_sex_effect_per_channel_regen.csv",
    },
    "theta": {
        "source": OUT_DIR / "R12345678_isc_by_band_theta.csv",
        "original": OUT_DIR / "R12345678_sex_effect_theta_per_channel.csv",
        "regen": OUT_DIR / "R12345678_sex_effect_theta_per_channel_regen.csv",
    },
}

COLUMNS = ["movie", "channel", "M_mean", "F_mean", "diff", "t", "p", "n_M", "n_F", "q_fdr"]


def build(source):
    isc = pd.read_csv(source)
    sex = pd.read_csv(MASTER)[["participant_id", "sex"]].drop_duplicates("participant_id")
    df = isc.merge(sex, on="participant_id", how="inner")
    ch_cols = [c for c in isc.columns if c.startswith("isc_E") or c == "isc_Cz"]

    rows = []
    for movie in MOVIE_ORDER:
        sub = df[df.movie == movie]
        m_all = sub[sub.sex == "M"]
        f_all = sub[sub.sex == "F"]
        movie_rows, movie_p = [], []
        for col in ch_cols:
            m = pd.to_numeric(m_all[col], errors="coerce").dropna()
            f = pd.to_numeric(f_all[col], errors="coerce").dropna()
            t, p = ttest_ind(m, f, equal_var=True)
            movie_rows.append({
                "movie": movie,
                "channel": col.replace("isc_", ""),
                "M_mean": float(m.mean()),
                "F_mean": float(f.mean()),
                "diff": float(m.mean() - f.mean()),
                "t": float(t),
                "p": float(p),
                "n_M": int(len(m)),
                "n_F": int(len(f)),
            })
            movie_p.append(float(p))
        # FDR within this movie only
        q = multipletests(movie_p, method="fdr_bh")[1]
        for r, qv in zip(movie_rows, q):
            r["q_fdr"] = float(qv)
        rows.extend(movie_rows)
    return pd.DataFrame(rows)[COLUMNS]


def verify(regen, original, tol=1e-9):
    if not original.exists():
        print(f"  original missing at {original}")
        return False
    orig = pd.read_csv(original)
    if len(orig) != len(regen):
        print(f"  ROW COUNT MISMATCH: original {len(orig)} vs regenerated {len(regen)}")
        return False
    key = ["movie", "channel"]
    a = regen.sort_values(key).reset_index(drop=True)
    b = orig.sort_values(key).reset_index(drop=True)
    if not a[key].equals(b[key]):
        print("  KEY MISMATCH: movie/channel pairs differ")
        return False
    ok = True
    for col in COLUMNS:
        if col in key:
            continue
        av = pd.to_numeric(a[col], errors="coerce").to_numpy(dtype=float)
        bv = pd.to_numeric(b[col], errors="coerce").to_numpy(dtype=float)
        both_nan = np.isnan(av) & np.isnan(bv)
        d = np.where(both_nan, 0.0, np.abs(av - bv))
        mx = float(np.nanmax(d))
        flag = "OK" if mx < tol else "MISMATCH"
        if mx >= tol:
            ok = False
            bad = int((d >= tol).sum())
            print(f"    {col:8s} max abs diff {mx:.3e}  {flag} ({bad} rows)")
        else:
            print(f"    {col:8s} max abs diff {mx:.3e}  {flag}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    all_ok = True
    for name, spec in SPECS.items():
        print(f"=== {name} ===")
        regen = build(spec["source"])
        regen.to_csv(spec["regen"], index=False)
        print(f"  built {len(regen)} rows, wrote {spec['regen'].name}")
        if args.verify:
            ok = verify(regen, spec["original"])
            all_ok = all_ok and ok
            print(f"  RESULT: {'MATCH' if ok else 'MISMATCH'}\n")
    if args.verify:
        print(f"OVERALL: {'ALL MATCH' if all_ok else 'MISMATCH PRESENT'}")


if __name__ == "__main__":
    main()
