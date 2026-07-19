"""Build the pooled master table across releases R1 to R8.

Reconstruction of outputs/R12345678_master.csv, which existed on disk with no
generating script. Verified to reproduce that file exactly.

The master joins, per subject:
  1. Per-subject aggregates of the movie preprocessing QC results
     (outputs/R{REL}_movies_results.csv, one row per subject and movie):
       n_movies_processed        number of movie rows for that subject
       n_movies_failed           count of rows with failed == True
       n_movies_pass             count of rows with qc_flag == "pass"
       n_movies_review           count of rows with qc_flag == "review"
       max_n_bads                worst-case bad-channel count
       min_ic_brain              worst-case brain IC count
       min_icalabel_median_conf  worst-case ICALabel confidence
       max_std_p50_uV            worst-case median-channel amplitude
     The worst-case metrics skip failed rows, which carry no metrics.
  2. A subject-level QC verdict derived from those counts:
       fail    if any movie failed to process
       review  else if any movie was flagged review
       pass    otherwise
     with subject_qc_reason left null for passing subjects.
  3. The release phenotype file (data/R{REL}_participants.tsv), supplying sex,
     age, EHQ, the four CBCL bifactor scores, and the per-task duration columns.

Usage:
    python scripts/build_master_table.py --verify
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"
ORIGINAL = OUT_DIR / "R12345678_master.csv"
REGEN = OUT_DIR / "R12345678_master_regen.csv"

RELEASES = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"]

QC_COLS = [
    "n_movies_processed", "n_movies_failed", "n_movies_pass", "n_movies_review",
    "max_n_bads", "min_ic_brain", "min_icalabel_median_conf", "max_std_p50_uV",
    "subject_qc_flag", "subject_qc_reason",
]


def aggregate_release(rel):
    """Per-subject QC aggregates for one release."""
    res = pd.read_csv(OUT_DIR / f"{rel}_movies_results.csv")
    rows = []
    for sid, g in res.groupby("participant_id", sort=False):
        failed = g["failed"].fillna(False).astype(bool)
        ok = g[~failed]
        n_proc = len(g)
        n_failed = int(failed.sum())
        n_pass = int((g["qc_flag"] == "pass").sum())
        n_review = int((g["qc_flag"] == "review").sum())
        if n_failed > 0:
            flag = "fail"
            reason = f"{n_failed} of {n_proc} movies failed"
        elif n_review > 0:
            flag = "review"
            reason = f"{n_review} movie(s) flagged review"
        else:
            flag = "pass"
            reason = np.nan
        rows.append({
            "participant_id": sid,
            "n_movies_processed": n_proc,
            "n_movies_failed": n_failed,
            "n_movies_pass": n_pass,
            "n_movies_review": n_review,
            "max_n_bads": ok["n_bads"].max() if len(ok) else np.nan,
            "min_ic_brain": ok["ic_brain"].min() if len(ok) else np.nan,
            "min_icalabel_median_conf": ok["icalabel_median_conf"].min() if len(ok) else np.nan,
            "max_std_p50_uV": ok["std_p50_uV"].max() if len(ok) else np.nan,
            "subject_qc_flag": flag,
            "subject_qc_reason": reason,
            "release": rel,
        })
    return pd.DataFrame(rows)


def build():
    parts = []
    for rel in RELEASES:
        qc = aggregate_release(rel)
        pheno = pd.read_csv(DATA_DIR / f"{rel}_participants.tsv", sep="\t")
        merged = qc.merge(pheno, on="participant_id", how="inner")
        parts.append(merged)
    out = pd.concat(parts, ignore_index=True)
    pheno_cols = [c for c in out.columns
                  if c not in QC_COLS + ["participant_id", "release"]]
    return out[["participant_id"] + QC_COLS + pheno_cols + ["release"]]


def verify(regen, tol=1e-9):
    if not ORIGINAL.exists():
        print(f"  original missing at {ORIGINAL}")
        return False
    orig = pd.read_csv(ORIGINAL)
    print(f"  original {orig.shape}  regenerated {regen.shape}")
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
        print("  COLUMN ORDER OR SET MISMATCH")
        print(f"    only in original   : {[c for c in orig.columns if c not in regen.columns]}")
        print(f"    only in regenerated: {[c for c in regen.columns if c not in orig.columns]}")
        return False
    a = regen.sort_values("participant_id").reset_index(drop=True)
    b = orig.sort_values("participant_id").reset_index(drop=True)
    ok = True
    for col in orig.columns:
        av, bv = a[col], b[col]
        if av.dtype.kind in "fi" and bv.dtype.kind in "fi":
            x = pd.to_numeric(av, errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(bv, errors="coerce").to_numpy(dtype=float)
            both_nan = np.isnan(x) & np.isnan(y)
            d = np.where(both_nan, 0.0, np.abs(x - y))
            n_bad = int((d > tol).sum())
        else:
            neq = (av.astype(str) != bv.astype(str)) & ~(av.isna() & bv.isna())
            n_bad = int(neq.sum())
        if n_bad:
            ok = False
            print(f"    DIFF {col}: {n_bad} rows")
            idx = np.where(
                (pd.to_numeric(av, errors="coerce").to_numpy(dtype=float)
                 != pd.to_numeric(bv, errors="coerce").to_numpy(dtype=float))
                if av.dtype.kind in "fi" and bv.dtype.kind in "fi"
                else ((av.astype(str) != bv.astype(str)) & ~(av.isna() & bv.isna())).to_numpy()
            )[0][:3]
            for i in idx:
                print(f"      row {i} ({a.participant_id.iloc[i]}): "
                      f"regen={av.iloc[i]!r} orig={bv.iloc[i]!r}")
    if ok:
        print(f"  all {len(orig.columns)} columns identical")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    regen = build()
    print(f"built master: n={len(regen)} subjects across {len(RELEASES)} releases")
    print(f"  qc flags: {regen.subject_qc_flag.value_counts().to_dict()}")
    regen.to_csv(REGEN, index=False)
    print(f"wrote {REGEN}")
    if args.verify:
        print("\nverification against the original:")
        ok = verify(regen)
        print(f"\nRESULT: {'MATCH' if ok else 'MISMATCH'}")


if __name__ == "__main__":
    main()
