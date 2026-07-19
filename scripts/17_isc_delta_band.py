"""Delta-band ISC for the ocular artifact validity gate (Gate 2).

The reported sex effect (male greater than female ISC) is theta peaked (4 to 8
Hz) and null in alpha and beta. The ocular hypothesis predicts the opposite
spectral profile: blink and saccade residual energy is maximal at low frequency
(delta, 1 to 4 Hz) and falls through theta. This script computes delta-band ISC
on the same balanced 400 subsample used for the published theta result, so the
delta versus theta sex-effect comparison is apples to apples.

Pipeline is identical to scripts/13_isc_by_band_full.py (three streaming passes,
leave-one-out Pearson r, template (sum_all - x_i) / (N - 1) per channel, float32
filter path). Theta is recomputed here purely as a reproduction gate: it must
match the published outputs/R12345678_isc_by_band_theta.csv before the delta
numbers from the same pipeline are trusted. Only the new delta CSV is written;
no existing file is overwritten.

Output: outputs/R12345678_isc_by_band_delta.csv (wide, one row per subject and
movie, isc_grand_mean plus per-channel ISC), mirroring the band CSV format.
"""
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
import mne

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBSAMPLE_CSV = PROJECT_ROOT / "data" / "R12345678_band_subsample.csv"
OUT = PROJECT_ROOT / "outputs"

MOVIES = ["DespicableMe", "DiaryOfAWimpyKid", "FunwithFractals", "ThePresent"]
# theta is the reproduction gate, delta is the new test
BANDS = {"theta": (4.0, 8.0), "delta": (1.0, 4.0)}
MIN_DUR_S = {"ThePresent": 200.0}
# theta uses the identical float32 code path as script 13, so it must reproduce
# the published theta CSV bit-for-bit. Gate at 1e-9 (effectively zero).
GATE_TOL = 1e-9


def fif_path(release, sid, movie):
    return OUT / f"preprocessed_movies_{release}" / \
        f"{sid}_{movie}_preproc_v3concat_raw.fif"


def read_raw_retry(path, preload, tries=6, wait=10.0):
    """Read a FIF, surviving transient USB SSD unmounts via retry plus a
    wait-for-remount loop. Mirrors the robustness used in the Test B pipeline."""
    last = None
    for k in range(tries):
        try:
            return mne.io.read_raw_fif(str(path), preload=preload, verbose="ERROR")
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"    read retry {k + 1}/{tries} for {Path(path).name}: {e}", flush=True)
            t_wait = time.time()
            while not Path(path).exists() and (time.time() - t_wait) < wait:
                time.sleep(1.0)
            time.sleep(min(wait, 3.0))
    raise last


def filter_to_band(data_f32, sfreq, lo, hi):
    """Zero-phase FIR bandpass via mne.filter.filter_data. float64 internally,
    returns float32. data_f32 shape (C, T). Identical to script 13."""
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
        raw = read_raw_retry(fif, preload=False)
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


def gate_theta(theta_rows, ch_names):
    """Compare recomputed theta wide rows against the published theta CSV.
    Returns max absolute difference over all isc columns plus isc_grand_mean."""
    pub_path = OUT / "R12345678_isc_by_band_theta.csv"
    pub = pd.read_csv(pub_path)
    new = pd.DataFrame(theta_rows)
    cols = ["isc_grand_mean"] + [f"isc_{c}" for c in ch_names]
    key = ["participant_id", "movie"]
    mg = pub[key + cols].merge(new[key + cols], on=key, suffixes=("_pub", "_new"),
                               how="outer", indicator=True)
    n_unmatched = int((mg["_merge"] != "both").sum())
    max_abs = 0.0
    for c in cols:
        a = pd.to_numeric(mg[f"{c}_pub"], errors="coerce").to_numpy()
        b = pd.to_numeric(mg[f"{c}_new"], errors="coerce").to_numpy()
        d = np.abs(a - b)
        both = np.isnan(a) & np.isnan(b)
        one = np.isnan(a) ^ np.isnan(b)
        d = np.where(both, 0.0, d)
        d = np.where(one, np.inf, d)
        if d.size:
            max_abs = max(max_abs, float(np.nanmax(d)))
    return max_abs, n_unmatched


def main():
    subsample = pd.read_csv(SUBSAMPLE_CSV)
    print(f"=== Delta-band ISC (Gate 2): subsample n={len(subsample)} ===", flush=True)
    print(f"  sex: {subsample.sex.value_counts().to_dict()}", flush=True)
    print(f"  bands: {BANDS} (theta is the reproduction gate, delta is the test)", flush=True)
    t_start = time.time()

    rows_per_band = {b: [] for b in BANDS}
    ch_names_ref_global = None

    for movie in MOVIES:
        t_movie = time.time()
        print(f"\n--- {movie} ---", flush=True)

        accepted, ch_names, sfreq, t_min, ss, sm, sx = pass1_inventory(subsample, movie)
        N = len(accepted)
        C = len(ch_names) if ch_names else 0
        print(f"  pass1: N={N} accepted (skipped {ss} short, {sm} missing, {sx} mismatch); "
              f"sfreq={sfreq}, T_min={t_min} ({t_min / sfreq:.1f}s)", flush=True)
        if N == 0:
            continue
        if ch_names_ref_global is None:
            ch_names_ref_global = ch_names

        # PASS 2: accumulate sum_all per band
        sum_all = {b: np.zeros((C, t_min), dtype=np.float32) for b in BANDS}
        ts = time.time()
        for sid, rel in accepted:
            raw = read_raw_retry(fif_path(rel, sid, movie), preload=True)
            x = raw.get_data().astype(np.float32)[:, :t_min]
            del raw
            for b, (lo, hi) in BANDS.items():
                xb = filter_to_band(x, sfreq, lo, hi)
                sum_all[b] += xb
                del xb
            del x
        print(f"  pass2 (sum, {len(BANDS)} bands) in {time.time() - ts:.1f}s", flush=True)

        # PASS 3: per-subject LOO Pearson r per band
        inv = 1.0 / (N - 1) if N > 1 else 1.0
        isc = {b: np.empty((N, C), dtype=np.float64) for b in BANDS}
        ts = time.time()
        for i, (sid, rel) in enumerate(accepted):
            raw = read_raw_retry(fif_path(rel, sid, movie), preload=True)
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
                print(f"    pass3 progress: {i + 1}/{N} subjects", flush=True)
        del sum_all
        print(f"  pass3 (LOO ISC, {len(BANDS)} bands) in {time.time() - ts:.1f}s", flush=True)

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
        print(f"  done in {time.time() - t_movie:.1f}s", flush=True)

    # Reproduction gate on theta before writing delta
    max_abs, n_unmatched = gate_theta(rows_per_band["theta"], ch_names_ref_global)
    verdict = "OK bit-identical" if max_abs == 0.0 else (
        "WITHIN FP" if max_abs < GATE_TOL else "FAIL")
    print(f"\n=== THETA REPRODUCTION GATE ===", flush=True)
    print(f"  vs published R12345678_isc_by_band_theta.csv: unmatched={n_unmatched} "
          f"max abs diff={max_abs:.3e} -> {verdict}", flush=True)
    if n_unmatched != 0 or max_abs >= GATE_TOL:
        print("  GATE FAILED. Delta CSV NOT written. STOP and report.", flush=True)
        raise SystemExit(1)

    # Write only the new delta CSV
    wide_fields = ["participant_id", "release", "movie", "band", "isc_grand_mean"] + \
                  [f"isc_{c}" for c in ch_names_ref_global]
    out = OUT / "R12345678_isc_by_band_delta.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=wide_fields, extrasaction="ignore")
        w.writeheader()
        for r in rows_per_band["delta"]:
            w.writerow(r)
    print(f"Wrote {out} ({len(rows_per_band['delta'])} rows)", flush=True)
    print(f"\nTotal wall: {(time.time() - t_start) / 60:.1f} min", flush=True)
    print("DELTA BAND ISC COMPLETE", flush=True)


if __name__ == "__main__":
    main()
