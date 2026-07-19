"""Patch: re-run ThePresent ISC with min-duration filter.

The first pass of scripts/11_compute_isc_R12345678.py used the global min-T
crop, which got dragged down to 57.8s by 4 outlier subjects with truncated
ThePresent recordings (3 in R8, 1 in R5). This script:

1. Identifies subjects with ThePresent duration >= 200s.
2. Recomputes ISC for ThePresent only on the filtered set.
3. Drops the bad ThePresent rows from R12345678_isc_per_subject.csv and
   R12345678_isc_long.csv and replaces them with the corrected rows.
"""
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
import mne

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COHORT_CSV = PROJECT_ROOT / "outputs" / "R12345678_isc_cohort.csv"
WIDE_CSV = PROJECT_ROOT / "outputs" / "R12345678_isc_per_subject.csv"
LONG_CSV = PROJECT_ROOT / "outputs" / "R12345678_isc_long.csv"

MOVIE = "ThePresent"
MIN_DUR_S = 200.0


def fif_path(release, sid):
    return PROJECT_ROOT / "outputs" / f"preprocessed_movies_{release}" / \
        f"{sid}_{MOVIE}_preproc_v3concat_raw.fif"


def main():
    cohort = pd.read_csv(COHORT_CSV)
    print(f"=== {MOVIE} patch: cohort n={len(cohort)} ===", flush=True)
    t_start = time.time()

    print(f"\nPass 1: header inventory + duration filter (>= {MIN_DUR_S}s) ...", flush=True)
    accepted = []
    ch_names_ref, sfreq_ref, t_min, skipped_short, skipped_missing = None, None, None, 0, 0
    for _, row in cohort.iterrows():
        sid, rel = row["participant_id"], row["release"]
        fif = fif_path(rel, sid)
        if not fif.exists():
            skipped_missing += 1
            continue
        raw = mne.io.read_raw_fif(str(fif), preload=False, verbose="ERROR")
        sf = float(raw.info["sfreq"])
        nT = raw.n_times
        dur = nT / sf
        if dur < MIN_DUR_S:
            skipped_short += 1
            continue
        names = list(raw.ch_names)
        if ch_names_ref is None:
            ch_names_ref = names
            sfreq_ref = sf
            t_min = nT
        else:
            if names != ch_names_ref or sf != sfreq_ref:
                skipped_short += 1
                continue
            if nT < t_min:
                t_min = nT
        accepted.append((sid, rel))
    N = len(accepted)
    print(f"  accepted={N} (skipped {skipped_short} short, {skipped_missing} missing); "
          f"sfreq={sfreq_ref}, T_min={t_min} samples ({t_min/sfreq_ref:.2f}s)", flush=True)

    print(f"\nPass 2: streaming sum_all ...", flush=True)
    sum_all = np.zeros((len(ch_names_ref), t_min), dtype=np.float32)
    for sid, rel in accepted:
        raw = mne.io.read_raw_fif(str(fif_path(rel, sid)), preload=True, verbose="ERROR")
        sum_all += raw.get_data().astype(np.float32)[:, :t_min]
        del raw
    print(f"  done in {time.time()-t_start:.1f}s", flush=True)

    print(f"\nPass 3: per-subject LOO Pearson ...", flush=True)
    inv = 1.0 / (N - 1)
    isc = np.empty((N, len(ch_names_ref)), dtype=np.float64)
    for i, (sid, rel) in enumerate(accepted):
        raw = mne.io.read_raw_fif(str(fif_path(rel, sid)), preload=True, verbose="ERROR")
        x = raw.get_data().astype(np.float32)[:, :t_min]
        loo = (sum_all - x) * inv
        a = x.astype(np.float64)
        b = loo.astype(np.float64)
        a_c = a - a.mean(axis=1, keepdims=True)
        b_c = b - b.mean(axis=1, keepdims=True)
        num = (a_c * b_c).sum(axis=1)
        den = np.sqrt((a_c ** 2).sum(axis=1) * (b_c ** 2).sum(axis=1))
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(den > 0, num / den, np.nan)
        isc[i, :] = r
        del raw, x, loo, a, b, a_c, b_c, num, den, r
    del sum_all

    grand = np.nanmean(isc, axis=1)
    print(f"  grand-mean ISC: mean={grand.mean():.4f} std={grand.std():.4f} "
          f"min={grand.min():.4f} max={grand.max():.4f}", flush=True)
    chan_mean = np.nanmean(isc, axis=0)
    top5 = np.argsort(-chan_mean)[:5]
    for j in top5:
        print(f"    top-channel {ch_names_ref[j]:>5}  ISC={chan_mean[j]:.4f}", flush=True)

    # Build new ThePresent rows
    new_wide = []
    new_long = []
    for i, (sid, rel) in enumerate(accepted):
        wrow = {"participant_id": sid, "release": rel, "movie": MOVIE,
                "isc_grand_mean": float(grand[i])}
        for j, ch in enumerate(ch_names_ref):
            wrow[f"isc_{ch}"] = float(isc[i, j]) if not np.isnan(isc[i, j]) else ""
        new_wide.append(wrow)
        for j, ch in enumerate(ch_names_ref):
            new_long.append({
                "participant_id": sid, "release": rel,
                "movie": MOVIE, "channel": ch,
                "isc": float(isc[i, j]) if not np.isnan(isc[i, j]) else "",
            })

    print(f"\nPatching {WIDE_CSV} ...", flush=True)
    wide_df = pd.read_csv(WIDE_CSV)
    print(f"  before: {len(wide_df)} rows; ThePresent rows: {(wide_df.movie == MOVIE).sum()}")
    wide_df = wide_df[wide_df.movie != MOVIE].copy()
    new_wide_df = pd.DataFrame(new_wide)
    # Align columns
    for c in wide_df.columns:
        if c not in new_wide_df.columns:
            new_wide_df[c] = ""
    new_wide_df = new_wide_df[wide_df.columns]
    wide_df = pd.concat([wide_df, new_wide_df], ignore_index=True)
    wide_df.to_csv(WIDE_CSV, index=False)
    print(f"  after: {len(wide_df)} rows; new ThePresent rows: {len(new_wide_df)}")

    print(f"\nPatching {LONG_CSV} ...", flush=True)
    long_df = pd.read_csv(LONG_CSV)
    print(f"  before: {len(long_df)} rows; ThePresent rows: {(long_df.movie == MOVIE).sum()}")
    long_df = long_df[long_df.movie != MOVIE].copy()
    new_long_df = pd.DataFrame(new_long)
    long_df = pd.concat([long_df, new_long_df], ignore_index=True)
    long_df.to_csv(LONG_CSV, index=False)
    print(f"  after: {len(long_df)} rows; new ThePresent rows: {len(new_long_df)}")

    print(f"\nTotal wall: {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
