"""Frequency-band ISC across R1-R8 (n=400 balanced subsample).

For each movie, performs 2 streaming passes over the FIFs; in each pass we
filter to all 3 bands (theta 4-8, alpha 8-12, beta 12-20) on the same loaded
data so each FIF is read only twice per movie regardless of band count.

Pass 1: header inventory + min-T determination (with min duration filter
        for ThePresent to drop the 4 truncated outliers).
Pass 2: stream-read each FIF, filter to 3 bands, accumulate sum_all per band.
Pass 3: stream-read each FIF, filter to 3 bands, per-subject LOO Pearson r
        per band per channel.

Outputs: outputs/R12345678_isc_by_band_{theta,alpha,beta}.csv -- wide,
one row per (subject, movie) with isc_grand_mean and per-channel ISC.
"""
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
import mne

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBSAMPLE_CSV = PROJECT_ROOT / "data" / "R12345678_band_subsample.csv"

MOVIES = ["DespicableMe", "DiaryOfAWimpyKid", "FunwithFractals", "ThePresent"]
BANDS = {"theta": (4.0, 8.0), "alpha": (8.0, 12.0), "beta": (12.0, 20.0)}
MIN_DUR_S = {"ThePresent": 200.0}


def fif_path(release, sid, movie):
    return PROJECT_ROOT / "outputs" / f"preprocessed_movies_{release}" / \
        f"{sid}_{movie}_preproc_v3concat_raw.fif"


def filter_to_band(data_f32, sfreq, lo, hi):
    """Zero-phase FIR bandpass via mne.filter.filter_data. Operates in-place
    style on float64 internally; returns float32. data_f32 shape (C, T)."""
    return mne.filter.filter_data(
        data_f32.astype(np.float64), sfreq, lo, hi, verbose="ERROR",
        method="fir", fir_design="firwin", phase="zero",
    ).astype(np.float32)


def pass1_inventory(subsample_df, movie):
    accepted = []
    ch_names_ref, sfreq_ref, t_min = None, None, None
    skipped_short, skipped_missing, skipped_mismatch = 0, 0, 0
    min_dur = MIN_DUR_S.get(movie, 0.0)
    for _, row in subsample_df.iterrows():
        sid, rel = row["participant_id"], row["release"]
        fif = fif_path(rel, sid, movie)
        if not fif.exists():
            skipped_missing += 1
            continue
        raw = mne.io.read_raw_fif(str(fif), preload=False, verbose="ERROR")
        sf = float(raw.info["sfreq"])
        nT = raw.n_times
        if (nT / sf) < min_dur:
            skipped_short += 1
            continue
        names = list(raw.ch_names)
        if ch_names_ref is None:
            ch_names_ref = names
            sfreq_ref = sf
            t_min = nT
        else:
            if names != ch_names_ref or sf != sfreq_ref:
                skipped_mismatch += 1
                continue
            if nT < t_min:
                t_min = nT
        accepted.append((sid, rel))
    return accepted, ch_names_ref, sfreq_ref, t_min, skipped_short, skipped_missing, skipped_mismatch


def main():
    subsample = pd.read_csv(SUBSAMPLE_CSV)
    print(f"=== Band ISC: subsample n={len(subsample)} ===", flush=True)
    print(f"  sex: {subsample.sex.value_counts().to_dict()}", flush=True)
    t_start = time.time()

    # We accumulate per-band wide rows
    rows_per_band = {b: [] for b in BANDS}
    ch_names_ref_global = None

    for movie in MOVIES:
        t_movie = time.time()
        print(f"\n--- {movie} ---", flush=True)

        accepted, ch_names, sfreq, t_min, ss, sm, sx = pass1_inventory(subsample, movie)
        N = len(accepted)
        C = len(ch_names) if ch_names else 0
        print(f"  pass1: N={N} accepted (skipped {ss} short, {sm} missing, {sx} mismatch); "
              f"sfreq={sfreq}, T_min={t_min} ({t_min/sfreq:.1f}s)", flush=True)
        if N == 0:
            continue
        if ch_names_ref_global is None:
            ch_names_ref_global = ch_names

        # PASS 2: accumulate sum_all per band
        sum_all = {b: np.zeros((C, t_min), dtype=np.float32) for b in BANDS}
        ts = time.time()
        for sid, rel in accepted:
            raw = mne.io.read_raw_fif(str(fif_path(rel, sid, movie)),
                                       preload=True, verbose="ERROR")
            x = raw.get_data().astype(np.float32)[:, :t_min]
            del raw
            for b, (lo, hi) in BANDS.items():
                xb = filter_to_band(x, sfreq, lo, hi)
                sum_all[b] += xb
                del xb
            del x
        print(f"  pass2 (sum, 3 bands) in {time.time()-ts:.1f}s", flush=True)

        # PASS 3: per-subject LOO Pearson r per band
        inv = 1.0 / (N - 1) if N > 1 else 1.0
        isc = {b: np.empty((N, C), dtype=np.float64) for b in BANDS}
        ts = time.time()
        for i, (sid, rel) in enumerate(accepted):
            raw = mne.io.read_raw_fif(str(fif_path(rel, sid, movie)),
                                       preload=True, verbose="ERROR")
            x = raw.get_data().astype(np.float32)[:, :t_min]
            del raw
            for b, (lo, hi) in BANDS.items():
                xb = filter_to_band(x, sfreq, lo, hi)
                loo = (sum_all[b] - xb) * inv
                a = xb.astype(np.float64)
                bv = loo.astype(np.float64)
                a_c = a - a.mean(axis=1, keepdims=True)
                b_c = bv - bv.mean(axis=1, keepdims=True)
                num = (a_c * b_c).sum(axis=1)
                den = np.sqrt((a_c ** 2).sum(axis=1) * (b_c ** 2).sum(axis=1))
                with np.errstate(divide="ignore", invalid="ignore"):
                    r = np.where(den > 0, num / den, np.nan)
                isc[b][i, :] = r
                del xb, loo, a, bv, a_c, b_c, num, den, r
            del x
            if (i + 1) % 50 == 0:
                print(f"    pass3 progress: {i+1}/{N} subjects", flush=True)
        del sum_all
        print(f"  pass3 (LOO ISC, 3 bands) in {time.time()-ts:.1f}s", flush=True)

        for b in BANDS:
            grand = np.nanmean(isc[b], axis=1)
            chan_mean = np.nanmean(isc[b], axis=0)
            top5 = np.argsort(-chan_mean)[:5]
            print(f"  {b}: grand-mean ISC mean={grand.mean():.4f} std={grand.std():.4f} "
                  f"top channels: {[ch_names[j] for j in top5]}", flush=True)
            for i, (sid, rel) in enumerate(accepted):
                row = {"participant_id": sid, "release": rel, "movie": movie,
                       "band": b, "isc_grand_mean": float(grand[i])}
                for j, ch in enumerate(ch_names):
                    row[f"isc_{ch}"] = float(isc[b][i, j]) if not np.isnan(isc[b][i, j]) else ""
                rows_per_band[b].append(row)
        del isc
        print(f"  done in {time.time()-t_movie:.1f}s", flush=True)

    # Write per-band CSVs
    wide_fields = ["participant_id","release","movie","band","isc_grand_mean"] + \
                  [f"isc_{c}" for c in ch_names_ref_global]
    for b in BANDS:
        out = PROJECT_ROOT / "outputs" / f"R12345678_isc_by_band_{b}.csv"
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=wide_fields, extrasaction="ignore")
            w.writeheader()
            for r in rows_per_band[b]:
                w.writerow(r)
        print(f"Wrote {out} ({len(rows_per_band[b])} rows)", flush=True)

    print(f"\nTotal wall: {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
