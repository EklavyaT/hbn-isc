"""Resting-state ISC sanity check on R1 (n=132).

Subjects record asynchronously during resting state, so leave-one-out ISC
should be ~0. Streaming 2-pass approach mirrors the band ISC pipeline:

Pass 1: header inventory + min-T crop.
Pass 2: stream-read each FIF, accumulate sum_all (broadband, no extra filter).
Pass 3: stream-read each FIF, per-subject LOO Pearson r per channel.

Output: outputs/R1_resting_state_isc.csv (one row per subject with
isc_grand_mean and per-channel ISC).
"""
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
import mne

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RS_DIR = PROJECT_ROOT / "outputs" / "preprocessed_R1"
OUT_CSV = PROJECT_ROOT / "outputs" / "R1_resting_state_isc.csv"


def main():
    fifs = sorted(RS_DIR.glob("*_RestingState_preproc_v3_raw.fif"))
    print(f"=== R1 Resting-State ISC: {len(fifs)} subjects ===", flush=True)
    t_start = time.time()

    # Pass 1: inventory
    accepted = []
    ch_names_ref = sfreq_ref = t_min = None
    for f in fifs:
        sid = f.name.split("_RestingState")[0]
        raw = mne.io.read_raw_fif(str(f), preload=False, verbose="ERROR")
        sf = float(raw.info["sfreq"])
        nT = raw.n_times
        names = list(raw.ch_names)
        if ch_names_ref is None:
            ch_names_ref, sfreq_ref, t_min = names, sf, nT
        else:
            if names != ch_names_ref or sf != sfreq_ref:
                continue
            t_min = min(t_min, nT)
        accepted.append((sid, f))
    N = len(accepted)
    C = len(ch_names_ref)
    print(f"pass1: N={N}, sfreq={sfreq_ref}, T_min={t_min} ({t_min/sfreq_ref:.1f}s)", flush=True)

    # Pass 2: sum_all
    ts = time.time()
    sum_all = np.zeros((C, t_min), dtype=np.float32)
    for sid, f in accepted:
        raw = mne.io.read_raw_fif(str(f), preload=True, verbose="ERROR")
        sum_all += raw.get_data().astype(np.float32)[:, :t_min]
        del raw
    print(f"pass2 sum_all in {time.time()-ts:.1f}s", flush=True)

    # Pass 3: per-subject LOO Pearson
    ts = time.time()
    inv = 1.0 / (N - 1)
    isc = np.empty((N, C), dtype=np.float64)
    for i, (sid, f) in enumerate(accepted):
        raw = mne.io.read_raw_fif(str(f), preload=True, verbose="ERROR")
        x = raw.get_data().astype(np.float32)[:, :t_min]
        del raw
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
        del x, loo, a, b, a_c, b_c, num, den, r
    print(f"pass3 LOO ISC in {time.time()-ts:.1f}s", flush=True)

    grand = np.nanmean(isc, axis=1)
    print(f"\nGrand-mean ISC: mean={grand.mean():+.4f} std={grand.std():.4f} "
          f"min={grand.min():+.4f} max={grand.max():+.4f}", flush=True)
    chan_mean = np.nanmean(isc, axis=0)
    top5 = np.argsort(-chan_mean)[:5]
    print("Top channels (by ISC):", flush=True)
    for j in top5:
        print(f"  {ch_names_ref[j]:>5}  ISC={chan_mean[j]:+.4f}", flush=True)

    fields = ["participant_id", "isc_grand_mean"] + [f"isc_{c}" for c in ch_names_ref]
    with OUT_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for i, (sid, _) in enumerate(accepted):
            row = {"participant_id": sid, "isc_grand_mean": float(grand[i])}
            for j, ch in enumerate(ch_names_ref):
                row[f"isc_{ch}"] = float(isc[i, j]) if not np.isnan(isc[i, j]) else ""
            w.writerow(row)
    print(f"\nWrote {OUT_CSV} ({N} rows)", flush=True)
    print(f"Total wall: {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
