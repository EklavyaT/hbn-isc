"""Pooled ISC across R1+R2+R3+R4+R5+R6+R7+R8.

Same algorithm as scripts/08_compute_isc_pooled.py (leave-one-out per-channel
Pearson r), but at 1100+ subjects the (N, C, T) float32 tensor would exceed
8 GB RAM. So this version streams: Pass 1 accumulates sum_all and the common
crop length T from disk; Pass 2 re-reads each FIF, computes per-subject LOO
mean = (sum_all - x_i) / (N-1) and Pearson r without ever holding the full
tensor in memory.

Outputs:
  outputs/R12345678_isc_per_subject.csv  -- wide: one row per (subject, movie)
  outputs/R12345678_isc_long.csv         -- long: one row per (subject, movie, channel)
"""
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
import mne

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COHORT_CSV = PROJECT_ROOT / "outputs" / "R12345678_isc_cohort.csv"
OUT_WIDE = PROJECT_ROOT / "outputs" / "R12345678_isc_per_subject.csv"
OUT_LONG = PROJECT_ROOT / "outputs" / "R12345678_isc_long.csv"

MOVIES = ["DespicableMe", "DiaryOfAWimpyKid", "FunwithFractals", "ThePresent"]


def fif_path(release, sid, movie):
    return PROJECT_ROOT / "outputs" / f"preprocessed_movies_{release}" / \
        f"{sid}_{movie}_preproc_v3concat_raw.fif"


def pass1_inventory(cohort_df, movie):
    """Pass 1: walk the cohort, find which (subject, release) pairs have a
    valid FIF and a consistent ch_names + sfreq. Return min duration and the
    accepted list."""
    accepted = []
    ch_names_ref = None
    sfreq_ref = None
    t_min = None
    skipped = 0
    for _, row in cohort_df.iterrows():
        sid = row["participant_id"]
        rel = row["release"]
        fif = fif_path(rel, sid, movie)
        if not fif.exists():
            skipped += 1
            continue
        # cheap header read
        raw = mne.io.read_raw_fif(str(fif), preload=False, verbose="ERROR")
        names = list(raw.ch_names)
        sf = float(raw.info["sfreq"])
        nT = raw.n_times
        if ch_names_ref is None:
            ch_names_ref = names
            sfreq_ref = sf
            t_min = nT
        else:
            if names != ch_names_ref or sf != sfreq_ref:
                skipped += 1
                continue
            if nT < t_min:
                t_min = nT
        accepted.append((sid, rel))
    return accepted, ch_names_ref, sfreq_ref, t_min, skipped


def pass2_sum(accepted, movie, t_min, n_channels):
    """Pass 2: stream-load each accepted FIF, accumulate sum_all (C, T) at
    float32. Returns sum_all."""
    sum_all = np.zeros((n_channels, t_min), dtype=np.float32)
    for sid, rel in accepted:
        raw = mne.io.read_raw_fif(str(fif_path(rel, sid, movie)),
                                   preload=True, verbose="ERROR")
        x = raw.get_data().astype(np.float32)[:, :t_min]
        sum_all += x
        del raw, x
    return sum_all


def pass3_isc(accepted, movie, t_min, sum_all):
    """Pass 3: for each subject, load FIF, compute LOO mean = (sum_all - x_i) / (N-1),
    then per-channel Pearson r. Return isc array (N, C) float64."""
    N, C = len(accepted), sum_all.shape[0]
    inv = 1.0 / (N - 1) if N > 1 else 1.0
    isc = np.empty((N, C), dtype=np.float64)
    for i, (sid, rel) in enumerate(accepted):
        raw = mne.io.read_raw_fif(str(fif_path(rel, sid, movie)),
                                   preload=True, verbose="ERROR")
        x = raw.get_data().astype(np.float32)[:, :t_min]
        loo = (sum_all - x) * inv  # (C, T) float32
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
    return isc


def main():
    cohort = pd.read_csv(COHORT_CSV)
    print(f"=== R1-R8 pooled ISC: cohort n={len(cohort)} ===", flush=True)
    print(f"  releases: {cohort.release.value_counts().sort_index().to_dict()}", flush=True)
    t_start = time.time()

    wide_rows = []
    long_rows = []
    ch_names_ref = None

    for movie in MOVIES:
        t0 = time.time()
        print(f"\n--- {movie} ---", flush=True)
        accepted, ch_names, sfreq, t_min, skipped = pass1_inventory(cohort, movie)
        n_use = len(accepted)
        print(f"  pass1: N={n_use} accepted (skipped {skipped}); "
              f"sfreq={sfreq}, T_min={t_min} samples ({t_min/sfreq:.1f}s)", flush=True)
        if ch_names_ref is None:
            ch_names_ref = ch_names
        else:
            assert ch_names == ch_names_ref, "channel order differs across movies"

        ts = time.time()
        sum_all = pass2_sum(accepted, movie, t_min, len(ch_names))
        print(f"  pass2 sum_all in {time.time()-ts:.1f}s", flush=True)

        ts = time.time()
        isc = pass3_isc(accepted, movie, t_min, sum_all)
        del sum_all
        print(f"  pass3 isc in {time.time()-ts:.1f}s", flush=True)

        grand = np.nanmean(isc, axis=1)
        print(f"  grand-mean ISC: mean={grand.mean():.4f} std={grand.std():.4f} "
              f"min={grand.min():.4f} max={grand.max():.4f}", flush=True)
        chan_mean = np.nanmean(isc, axis=0)
        top5 = np.argsort(-chan_mean)[:5]
        for j in top5:
            print(f"    top-channel {ch_names[j]:>5}  ISC={chan_mean[j]:.4f}", flush=True)

        for i, (sid, rel) in enumerate(accepted):
            row = {"participant_id": sid, "release": rel, "movie": movie,
                   "isc_grand_mean": float(grand[i])}
            for j, ch in enumerate(ch_names):
                row[f"isc_{ch}"] = float(isc[i, j]) if not np.isnan(isc[i, j]) else ""
            wide_rows.append(row)
            for j, ch in enumerate(ch_names):
                long_rows.append({
                    "participant_id": sid, "release": rel,
                    "movie": movie, "channel": ch,
                    "isc": float(isc[i, j]) if not np.isnan(isc[i, j]) else "",
                })
        del isc
        print(f"  done in {time.time()-t0:.1f}s", flush=True)

    wide_fields = ["participant_id", "release", "movie", "isc_grand_mean"] + \
                  [f"isc_{c}" for c in ch_names_ref]
    OUT_WIDE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_WIDE.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=wide_fields, extrasaction="ignore")
        w.writeheader()
        for r in wide_rows:
            w.writerow(r)
    print(f"\nWrote {OUT_WIDE} ({len(wide_rows)} rows)", flush=True)

    with OUT_LONG.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["participant_id", "release", "movie", "channel", "isc"])
        w.writeheader()
        for r in long_rows:
            w.writerow(r)
    print(f"Wrote {OUT_LONG} ({len(long_rows)} rows)", flush=True)
    print(f"\nTotal wall clock: {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
