"""Build the ISC cohort (n=1143) from the master table.

Reconstruction of outputs/R12345678_isc_cohort.csv, which existed on disk with
no generating script. Verified to reproduce that file exactly.

The cohort is the set of subjects eligible for the ISC analyses. Selection
criteria, applied to outputs/R12345678_master.csv (n=1535):

  1. Loose QC thresholds on the worst-case-across-movies metrics:
       max_std_p50_uV            <= 40
       min_icalabel_median_conf  >= 0.50
       min_ic_brain              >= 3
  2. All four movies processed successfully (n_movies_failed == 0) and all four
     passing QC (n_movies_pass == 4), with a non-null duration entry for each of
     the four movie columns.
  3. Complete CBCL bifactor scores (p_factor, attention, internalizing,
     externalizing all non-null).

Note on criterion 2. Filtering on n_movies_pass == 4 alone yields 1145 subjects,
two more than the stored cohort. Two subjects (sub-NDARCU736GZ1 and
sub-NDARPE752VYE) had movies that FAILED to process while every movie that did
process passed QC, so they carry n_movies_pass == 4 but were excluded. Requiring
n_movies_failed == 0 reproduces the stored cohort exactly. The master table's
own subject_qc_flag column already encodes the full rule, and the explicit
criteria above were confirmed to select an identical set of subjects.

Usage:
    python scripts/build_isc_cohort.py --verify
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"
MASTER = OUT_DIR / "R12345678_master.csv"
ORIGINAL = OUT_DIR / "R12345678_isc_cohort.csv"
REGEN = OUT_DIR / "R12345678_isc_cohort_regen.csv"

MOVIES = ["DespicableMe", "FunwithFractals", "ThePresent", "DiaryOfAWimpyKid"]
CBCL = ["p_factor", "attention", "internalizing", "externalizing"]

STD_MAX = 40.0
CONF_MIN = 0.50
BRAIN_MIN = 3


def build():
    m = pd.read_csv(MASTER)
    keep = (
        (m.max_std_p50_uV <= STD_MAX)
        & (m.min_icalabel_median_conf >= CONF_MIN)
        & (m.min_ic_brain >= BRAIN_MIN)
        & (m.n_movies_pass == 4)
        & (m.n_movies_failed == 0)
        & (m[MOVIES].notna().all(axis=1))
        & (m[CBCL].notna().all(axis=1))
    )
    return m[keep].reset_index(drop=True)


def verify(regen, tol=1e-9):
    if not ORIGINAL.exists():
        print(f"  original missing at {ORIGINAL}")
        return False
    orig = pd.read_csv(ORIGINAL)
    print(f"  original n={len(orig)}  regenerated n={len(regen)}")
    if set(regen.participant_id) != set(orig.participant_id):
        only_r = set(regen.participant_id) - set(orig.participant_id)
        only_o = set(orig.participant_id) - set(regen.participant_id)
        print(f"  SUBJECT SET MISMATCH: {len(only_r)} extra, {len(only_o)} missing")
        for s in list(only_r)[:5]:
            print(f"    extra:   {s}")
        for s in list(only_o)[:5]:
            print(f"    missing: {s}")
        return False
    print("  subject set identical: True")
    if list(regen.columns) != list(orig.columns):
        print("  COLUMN MISMATCH")
        return False
    a = regen.sort_values("participant_id").reset_index(drop=True)
    b = orig.sort_values("participant_id").reset_index(drop=True)
    ok = True
    for col in orig.columns:
        av, bv = a[col], b[col]
        if av.dtype.kind in "fi" and bv.dtype.kind in "fi":
            d = np.abs(pd.to_numeric(av, errors="coerce") - pd.to_numeric(bv, errors="coerce"))
            n_bad = int((d > tol).sum())
        else:
            neq = (av.astype(str) != bv.astype(str)) & ~(av.isna() & bv.isna())
            n_bad = int(neq.sum())
        if n_bad:
            ok = False
            print(f"    DIFF {col}: {n_bad} rows")
    if ok:
        print("  all 35 columns identical")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    regen = build()
    print(f"built cohort: n={len(regen)} "
          f"({int((regen.sex == 'M').sum())} male, {int((regen.sex == 'F').sum())} female)")
    regen.to_csv(REGEN, index=False)
    print(f"wrote {REGEN}")
    if args.verify:
        print("\nverification against the original:")
        ok = verify(regen)
        print(f"\nRESULT: {'MATCH' if ok else 'MISMATCH'}")


if __name__ == "__main__":
    main()
