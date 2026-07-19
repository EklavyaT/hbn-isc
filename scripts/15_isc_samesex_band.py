"""Same-sex template band ISC (Test A) on the n=400 balanced subsample.

Validity gate for Dr. Fu's concern: build the leave-one-out ISC template from
SAME-SEX subjects only (each male against the mean of other males, each female
against the mean of other females) and check whether the male-greater-than-female
frontocentral theta effect survives.

This script adapts scripts/13_isc_by_band_full.py. It keeps the three-pass
streaming structure and the EXACT Pearson computation, so the pooled output
reproduces 13 to floating-point tolerance. Per movie it computes BOTH:
  - the pooled LOO ISC, template (sum_all - x) / (N - 1)        [reproduction]
  - the same-sex LOO ISC, males vs (sum_M - x) / (N_M - 1),
    females vs (sum_F - x) / (N_F - 1)                          [Test A]

Because the subsample is balanced (about 200 male and 200 female), the same-sex
templates are already size-matched, so no subsampling is done here.

Outputs (NEW files, originals untouched):
  outputs/repro_R12345678_isc_by_band_{theta,alpha,beta}.csv   [pooled, repro]
  outputs/R12345678_isc_by_band_{theta,alpha,beta}_samesex.csv [same-sex]
Both use the identical wide schema to the originals.
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
    """Zero-phase FIR bandpass via mne.filter.filter_data. Identical to 13."""
    return mne.filter.filter_data(
        data_f32.astype(np.float64), sfreq, lo, hi, verbose="ERROR",
        method="fir", fir_design="firwin", phase="zero",
    ).astype(np.float32)


def pearson_loo(xb, loo):
    """Per-channel Pearson r between band signal xb and LOO template loo.

    Byte-for-byte the same arithmetic as scripts/13_isc_by_band_full.py
    (pass 3): cast both to float64, mean-center along time, then r = num/den.
    """
    a = xb.astype(np.float64)
    bv = loo.astype(np.float64)
    a_c = a - a.mean(axis=1, keepdims=True)
    b_c = bv - bv.mean(axis=1, keepdims=True)
    num = (a_c * b_c).sum(axis=1)
    den = np.sqrt((a_c ** 2).sum(axis=1) * (b_c ** 2).sum(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(den > 0, num / den, np.nan)
    return r


def pass1_inventory(subsample_df, movie):
    """Same as 13, but carries each accepted subject's sex through the list."""
    accepted = []
    ch_names_ref, sfreq_ref, t_min = None, None, None
    skipped_short, skipped_missing, skipped_mismatch = 0, 0, 0
    min_dur = MIN_DUR_S.get(movie, 0.0)
    for _, row in subsample_df.iterrows():
        sid, rel, sex = row["participant_id"], row["release"], row["sex"]
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
        accepted.append((sid, rel, sex))
    return accepted, ch_names_ref, sfreq_ref, t_min, skipped_short, skipped_missing, skipped_mismatch


def main():
    subsample = pd.read_csv(SUBSAMPLE_CSV)
    print(f"=== Same-sex band ISC: subsample n={len(subsample)} ===", flush=True)
    print(f"  sex: {subsample.sex.value_counts().to_dict()}", flush=True)
    t_start = time.time()

    rows_repro = {b: [] for b in BANDS}
    rows_samesex = {b: [] for b in BANDS}
    ch_names_ref_global = None

    for movie in MOVIES:
        t_movie = time.time()
        print(f"\n--- {movie} ---", flush=True)

        accepted, ch_names, sfreq, t_min, ss, sm, sx = pass1_inventory(subsample, movie)
        N = len(accepted)
        C = len(ch_names) if ch_names else 0
        sexes = np.array([s for (_, _, s) in accepted])
        N_M = int((sexes == "M").sum())
        N_F = int((sexes == "F").sum())
        print(f"  pass1: N={N} accepted (N_M={N_M}, N_F={N_F}; skipped {ss} short, "
              f"{sm} missing, {sx} mismatch); sfreq={sfreq}, T_min={t_min} "
              f"({t_min/sfreq:.1f}s)", flush=True)
        if N == 0:
            continue
        if ch_names_ref_global is None:
            ch_names_ref_global = ch_names

        is_male = (sexes == "M")
        is_female = (sexes == "F")

        # PASS 2: accumulate sum_all (pooled), sum_M, sum_F per band.
        # sum_all is accumulated in the exact accepted order used by 13, so the
        # pooled template is bit-identical to 13.
        sum_all = {b: np.zeros((C, t_min), dtype=np.float32) for b in BANDS}
        sum_M = {b: np.zeros((C, t_min), dtype=np.float32) for b in BANDS}
        sum_F = {b: np.zeros((C, t_min), dtype=np.float32) for b in BANDS}
        ts = time.time()
        for (sid, rel, sex) in accepted:
            raw = mne.io.read_raw_fif(str(fif_path(rel, sid, movie)),
                                       preload=True, verbose="ERROR")
            x = raw.get_data().astype(np.float32)[:, :t_min]
            del raw
            for b, (lo, hi) in BANDS.items():
                xb = filter_to_band(x, sfreq, lo, hi)
                sum_all[b] += xb
                if sex == "M":
                    sum_M[b] += xb
                else:
                    sum_F[b] += xb
                del xb
            del x
        print(f"  pass2 (sum_all + sum_M + sum_F, 3 bands) in {time.time()-ts:.1f}s",
              flush=True)

        # PASS 3: per subject LOO Pearson r per band, pooled and same-sex.
        inv = 1.0 / (N - 1) if N > 1 else 1.0
        inv_M = 1.0 / (N_M - 1) if N_M > 1 else 1.0
        inv_F = 1.0 / (N_F - 1) if N_F > 1 else 1.0
        isc_pool = {b: np.empty((N, C), dtype=np.float64) for b in BANDS}
        isc_ss = {b: np.empty((N, C), dtype=np.float64) for b in BANDS}
        ts = time.time()
        for i, (sid, rel, sex) in enumerate(accepted):
            raw = mne.io.read_raw_fif(str(fif_path(rel, sid, movie)),
                                       preload=True, verbose="ERROR")
            x = raw.get_data().astype(np.float32)[:, :t_min]
            del raw
            for b, (lo, hi) in BANDS.items():
                xb = filter_to_band(x, sfreq, lo, hi)
                # pooled template, identical to 13
                loo = (sum_all[b] - xb) * inv
                isc_pool[b][i, :] = pearson_loo(xb, loo)
                # same-sex template
                if sex == "M":
                    loo_ss = (sum_M[b] - xb) * inv_M
                else:
                    loo_ss = (sum_F[b] - xb) * inv_F
                isc_ss[b][i, :] = pearson_loo(xb, loo_ss)
                del xb, loo, loo_ss
            del x
            if (i + 1) % 50 == 0:
                print(f"    pass3 progress: {i+1}/{N} subjects", flush=True)
        del sum_all, sum_M, sum_F
        print(f"  pass3 (pooled + same-sex LOO ISC, 3 bands) in {time.time()-ts:.1f}s",
              flush=True)

        for b in BANDS:
            grand_p = np.nanmean(isc_pool[b], axis=1)
            grand_s = np.nanmean(isc_ss[b], axis=1)
            # quick frontocentral cluster glance for males vs females
            fc_idx = [ch_names.index(c) for c in ("E30", "E36", "E37") if c in ch_names]
            fc_p = np.nanmean(isc_pool[b][:, fc_idx], axis=1)
            fc_s = np.nanmean(isc_ss[b][:, fc_idx], axis=1)
            print(f"  {b}: pooled grand mean={grand_p.mean():.4f}  "
                  f"same-sex grand mean={grand_s.mean():.4f}", flush=True)
            print(f"      {b} frontocentral cluster (E30/E36/E37): "
                  f"pooled M={fc_p[is_male].mean():.4f} F={fc_p[is_female].mean():.4f} | "
                  f"same-sex M={fc_s[is_male].mean():.4f} F={fc_s[is_female].mean():.4f}",
                  flush=True)
            for i, (sid, rel, sex) in enumerate(accepted):
                row_p = {"participant_id": sid, "release": rel, "movie": movie,
                         "band": b, "isc_grand_mean": float(grand_p[i])}
                row_s = {"participant_id": sid, "release": rel, "movie": movie,
                         "band": b, "isc_grand_mean": float(grand_s[i])}
                for j, ch in enumerate(ch_names):
                    vp = isc_pool[b][i, j]
                    vs = isc_ss[b][i, j]
                    row_p[f"isc_{ch}"] = float(vp) if not np.isnan(vp) else ""
                    row_s[f"isc_{ch}"] = float(vs) if not np.isnan(vs) else ""
                rows_repro[b].append(row_p)
                rows_samesex[b].append(row_s)
        del isc_pool, isc_ss
        print(f"  done in {time.time()-t_movie:.1f}s", flush=True)

    wide_fields = ["participant_id", "release", "movie", "band", "isc_grand_mean"] + \
                  [f"isc_{c}" for c in ch_names_ref_global]
    for b in BANDS:
        out_repro = PROJECT_ROOT / "outputs" / f"repro_R12345678_isc_by_band_{b}.csv"
        out_ss = PROJECT_ROOT / "outputs" / f"R12345678_isc_by_band_{b}_samesex.csv"
        with out_repro.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=wide_fields, extrasaction="ignore")
            w.writeheader()
            for r in rows_repro[b]:
                w.writerow(r)
        with out_ss.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=wide_fields, extrasaction="ignore")
            w.writeheader()
            for r in rows_samesex[b]:
                w.writerow(r)
        print(f"Wrote {out_repro} ({len(rows_repro[b])} rows)", flush=True)
        print(f"Wrote {out_ss} ({len(rows_samesex[b])} rows)", flush=True)

    print(f"\nTotal wall: {(time.time()-t_start)/60:.1f} min", flush=True)
    print("STEP 2 COMPLETE", flush=True)


if __name__ == "__main__":
    main()
