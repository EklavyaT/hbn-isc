"""Build the primary regression analysis frame.

Reconstruction of outputs/R12345678_sex_analysis_df.csv, which existed on disk
with no generating script. Verified to reproduce that file exactly.

This is the frame every regression in the manuscript is fit on. It is the
per-subject per-movie broadband ISC joined to the subject-level covariates
carried by the ISC cohort:

  from outputs/R12345678_isc_per_subject.csv : participant_id, release, movie,
                                               isc_grand_mean
  from outputs/R12345678_isc_cohort.csv      : age, sex, ehq_total,
                                               release_number, the four CBCL
                                               bifactor scores, and the three
                                               worst-case QC metrics plus the
                                               QC flag

The join is an inner join on participant_id, so only cohort subjects survive,
giving 4568 rows (1143 subjects times 4 movies, minus the ThePresent recordings
dropped by the script 12 minimum-duration patch).

Usage:
    python scripts/build_sex_analysis_df.py --verify
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"
ISC = OUT_DIR / "R12345678_isc_per_subject.csv"
COHORT = OUT_DIR / "R12345678_isc_cohort.csv"
ORIGINAL = OUT_DIR / "R12345678_sex_analysis_df.csv"
REGEN = OUT_DIR / "R12345678_sex_analysis_df_regen.csv"

COVARIATES = [
    "participant_id", "age", "sex", "ehq_total", "release_number",
    "p_factor", "attention", "internalizing", "externalizing",
    "subject_qc_flag", "max_std_p50_uV", "min_icalabel_median_conf", "min_ic_brain",
]
COLUMNS = [
    "participant_id", "release", "movie", "isc_grand_mean", "age", "sex",
    "ehq_total", "release_number", "p_factor", "attention", "internalizing",
    "externalizing", "subject_qc_flag", "max_std_p50_uV",
    "min_icalabel_median_conf", "min_ic_brain",
]


def build():
    isc = pd.read_csv(ISC)[["participant_id", "release", "movie", "isc_grand_mean"]]
    cohort = pd.read_csv(COHORT)[COVARIATES].drop_duplicates("participant_id")
    return isc.merge(cohort, on="participant_id", how="inner")[COLUMNS]


def verify(regen, tol=1e-9):
    if not ORIGINAL.exists():
        print(f"  original missing at {ORIGINAL}")
        return False
    orig = pd.read_csv(ORIGINAL)
    print(f"  original {orig.shape}  regenerated {regen.shape}")
    if list(regen.columns) != list(orig.columns):
        print(f"  COLUMN MISMATCH\n    original    : {list(orig.columns)}"
              f"\n    regenerated : {list(regen.columns)}")
        return False
    key = ["participant_id", "movie"]
    if len(regen) != len(orig):
        print(f"  ROW COUNT MISMATCH: {len(orig)} vs {len(regen)}")
        return False
    a = regen.sort_values(key).reset_index(drop=True)
    b = orig.sort_values(key).reset_index(drop=True)
    if not a[key].astype(str).equals(b[key].astype(str)):
        print("  KEY MISMATCH: participant/movie pairs differ")
        return False
    ok = True
    for col in orig.columns:
        av, bv = a[col], b[col]
        if av.dtype.kind in "fi" and bv.dtype.kind in "fi":
            x = pd.to_numeric(av, errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(bv, errors="coerce").to_numpy(dtype=float)
            both_nan = np.isnan(x) & np.isnan(y)
            d = np.where(both_nan, 0.0, np.abs(x - y))
            n_bad = int((d > tol).sum())
            detail = f"max abs diff {np.nanmax(d):.3e}"
        else:
            neq = (av.astype(str) != bv.astype(str)) & ~(av.isna() & bv.isna())
            n_bad = int(neq.sum())
            detail = ""
        if n_bad:
            ok = False
            print(f"    DIFF {col}: {n_bad} rows {detail}")
    if ok:
        print(f"  all {len(orig.columns)} columns identical")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    regen = build()
    print(f"built analysis frame: {regen.shape[0]} rows, "
          f"{regen.participant_id.nunique()} subjects")
    regen.to_csv(REGEN, index=False)
    print(f"wrote {REGEN}")
    if args.verify:
        print("\nverification against the original:")
        ok = verify(regen)
        print(f"\nRESULT: {'MATCH' if ok else 'MISMATCH'}")


if __name__ == "__main__":
    main()
