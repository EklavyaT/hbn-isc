"""Build the balanced 400 subject subsample used for all frequency-band analyses.

Reconstruction of data/R12345678_band_subsample.csv, which existed on disk with
no generating script. Verified to reproduce that file exactly (all 400 subjects
identical). See REPO_AUDIT.md.

Construction: from the ISC cohort (outputs/R12345678_isc_cohort.csv, n=1143,
765 male and 378 female), draw 200 male and 200 female with pandas
DataFrame.sample(n=200, random_state=42) applied independently within each sex,
then sort by participant_id.

Two details are load bearing for exact reproduction:
  1. The pool must be the cohort file in its NATIVE row order, not sorted first.
     Sorting the pool before sampling changes which subjects are drawn.
  2. The random_state is applied per sex, not once for the whole draw. pandas
     .sample uses the legacy numpy RandomState, so this is equivalent to calling
     np.random.seed(42) immediately before each per-sex draw.

Usage:
    python scripts/build_band_subsample.py            # writes the _regen file
    python scripts/build_band_subsample.py --verify   # also diffs vs the original
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COHORT_CSV = PROJECT_ROOT / "outputs" / "R12345678_isc_cohort.csv"
ORIGINAL = PROJECT_ROOT / "data" / "R12345678_band_subsample.csv"
OUT = PROJECT_ROOT / "data" / "R12345678_band_subsample_regen.csv"

N_PER_SEX = 200
SEED = 42
COLUMNS = ["participant_id", "release", "sex", "age"]


def build():
    pool = pd.read_csv(COHORT_CSV)
    parts = []
    for sex in ["M", "F"]:
        sub = pool[pool.sex == sex]
        parts.append(sub.sample(n=N_PER_SEX, random_state=SEED))
    out = pd.concat(parts)[COLUMNS]
    return out.sort_values("participant_id").reset_index(drop=True)


def verify(regen):
    if not ORIGINAL.exists():
        print(f"  original not found at {ORIGINAL}, skipping verification")
        return None
    orig = pd.read_csv(ORIGINAL)
    same_ids = set(regen.participant_id) == set(orig.participant_id)
    n_overlap = len(set(regen.participant_id) & set(orig.participant_id))
    print(f"  original n={len(orig)}  regenerated n={len(regen)}")
    print(f"  subject overlap: {n_overlap}/{len(orig)}")
    print(f"  identical subject set: {same_ids}")
    if same_ids and len(regen) == len(orig):
        aligned = regen[COLUMNS].reset_index(drop=True)
        o = orig[COLUMNS].reset_index(drop=True)
        exact = aligned.equals(o)
        print(f"  full frame identical (all columns, row order): {exact}")
        if not exact:
            for col in COLUMNS:
                diff = (aligned[col] != o[col]) & ~(aligned[col].isna() & o[col].isna())
                if diff.any():
                    print(f"    column {col}: {int(diff.sum())} differing rows")
        return exact
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="diff the regenerated file against the original")
    args = ap.parse_args()

    regen = build()
    print(f"built balanced subsample: n={len(regen)} "
          f"({int((regen.sex == 'M').sum())} male, {int((regen.sex == 'F').sum())} female)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    regen.to_csv(OUT, index=False)
    print(f"wrote {OUT}")
    if args.verify:
        print("\nverification against the original:")
        ok = verify(regen)
        print(f"\nRESULT: {'MATCH' if ok else 'MISMATCH'}")


if __name__ == "__main__":
    main()
