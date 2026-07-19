"""Gate 2 Tier 2 re-fit module: re-fit ICA per subject and cache it, so the
ocular (eye-blink) components can be removed or retained at apply time to test
whether ocular-component removal creates or carries the frontocentral sex effect.

This REUSES the published pipeline scripts/preprocess_movies_v3_concat.py. The
per-movie pre-ICA preparation is copied verbatim from prepare_one_movie there (so
the load, montage, crop, notch, dual filter, CAR, PyPREP NoisyChannels with
random_state=42, and interpolate are identical), with two differences that are
the only permitted modifications for this run:
  1. The function also returns the per-movie bad-channel list, so STEP 4 can
     regenerate raw_analysis cheaply from the cached bads without re-running the
     expensive PyPREP NoisyChannels search. This exposes internal state only; it
     does not change the pipeline.
  2. The original fits ICA on raw_ica at the NATIVE 500 Hz and resamples to 200
     Hz only at the very end, after ica.apply. For tractability over 150 subjects
     this re-fit resamples raw_ica to 200 Hz BEFORE the ICA fit. This is a
     deliberate deviation, documented here and printed in the log; it is why the
     STEP 2 faithfulness correlation is expected to be high but not identity.

Cache per subject on the SSD (gate2_cache/):
  {sid}-ica.fif        fitted ICA (200 Hz)
  {sid}_gate2.json     ICALabel labels and probs, reject and eye-blink IC indices,
                       per-movie bad channels, per-movie crop sample counts, meta
The module is RESUMABLE: a subject whose cache is complete is skipped.

CLI:
  python scripts/18_gate2_refit.py --mode fit --all
  python scripts/18_gate2_refit.py --mode fit --sids sub-XXXX [sub-YYYY ...]
  python scripts/18_gate2_refit.py --mode faithfulness --n 5
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import mne

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
# reuse the published pipeline helpers and constants verbatim
import preprocess_movies_v3_concat as pp  # noqa: E402
from preprocess_movies_v3_concat import (  # noqa: E402
    find_eeg_files, parse_movie_window, band_power, NOTCH_FREQS, REJECT_LABELS,
)
from mne_icalabel import label_components  # noqa: E402
from pyprep.find_noisy_channels import NoisyChannels  # noqa: E402

MOVIES = ["DespicableMe", "DiaryOfAWimpyKid", "FunwithFractals", "ThePresent"]
MIN_DUR_S = {"ThePresent": 200.0}
ICA_SFREQ = 200.0  # re-fit rate (see modification 2)
SUBSAMPLE_CSV = PROJECT_ROOT / "data" / "R12345678_band_subsample.csv"
BAL150_CSV = PROJECT_ROOT / "data" / "gate2_balanced150.csv"
STORED_DIR = PROJECT_ROOT / "outputs"  # preprocessed_movies_R{rel} live here via symlink

# Cache for the per-subject ICA re-fits and intermediates. This grows to roughly
# a few GB across the 150 subject subsample and is what makes the run resumable,
# so it should live somewhere persistent with room to spare. Override with the
# environment variable HBN_GATE2_CACHE; defaults to ./outputs/gate2_cache under
# the project root. The original analysis pointed this at an external SSD at
# /Volumes/PortableSSD/Projects/HBN-BrainAI/outputs/gate2_cache.
CACHE_DIR = Path(os.environ.get("HBN_GATE2_CACHE",
                                str(PROJECT_ROOT / "outputs" / "gate2_cache")))

# electrode groups (reuse the exact Tier 1 sets, do not redefine)
FRONTOCENTRAL = ["E30", "E36", "E37"]
PERIOCULAR = ["E126", "E127", "E25", "E8", "E17", "E21", "E14", "E1", "E32", "E125"]


# ------------------------------------------------------------------ cache io

def ica_path(sid):
    return CACHE_DIR / f"{sid}-ica.fif"


def json_path(sid):
    return CACHE_DIR / f"{sid}_gate2.json"


def cache_complete(sid):
    if not ica_path(sid).exists() or not json_path(sid).exists():
        return False
    try:
        with json_path(sid).open() as f:
            meta = json.load(f)
        return bool(meta.get("complete", False))
    except Exception:
        return False


def retry(fn, tries=6, wait=8.0, what=""):
    """Run fn, surviving transient USB SSD unmounts via retry plus remount wait."""
    last = None
    for k in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"    SSD retry {k + 1}/{tries} ({what}): {e}", flush=True)
            t0 = time.time()
            while not CACHE_DIR.parent.exists() and (time.time() - t0) < wait:
                time.sleep(1.0)
            time.sleep(min(wait, 3.0))
    raise last


# ------------------------------------------------------------------ subsample

def build_balanced_150():
    sub = pd.read_csv(SUBSAMPLE_CSV).sort_values("participant_id").reset_index(drop=True)
    m = sub[sub.sex == "M"].head(75)
    f = sub[sub.sex == "F"].head(75)
    bal = pd.concat([m, f]).sort_values("participant_id").reset_index(drop=True)
    if not BAL150_CSV.exists():
        bal.to_csv(BAL150_CSV, index=False)
    return bal


# ------------------------------------------- verbatim pre-ICA prep (exposes bads)

def prepare_one_movie_capture(subject_id, task, set_path, events_path, verbose=True):
    """Verbatim copy of preprocess_movies_v3_concat.prepare_one_movie, returning
    the bad-channel list in addition to the usual triple. No pipeline logic is
    changed; only the return signature exposes `bads`."""
    metrics = {"participant_id": subject_id, "movie": task, "failed": False, "error": ""}

    raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose="WARNING")
    metrics["orig_duration_s"] = round(float(raw.times[-1]), 2)
    metrics["orig_n_channels"] = len(raw.ch_names)
    metrics["orig_sfreq"] = float(raw.info["sfreq"])

    raw.set_montage(
        mne.channels.make_standard_montage("GSN-HydroCel-129"),
        on_missing="warn",
        match_case=False,
    )

    v_start, v_stop = parse_movie_window(events_path)
    metrics["video_start_s"] = v_start
    metrics["video_stop_s"] = v_stop
    if v_start is None or v_stop is None:
        v_start, v_stop = 0.0, float(raw.times[-1])
    metrics["movie_duration_s"] = round(v_stop - v_start, 2)
    raw.crop(tmin=v_start, tmax=min(v_stop, float(raw.times[-1])))

    p_alpha_pre = band_power(raw, 8, 13)
    p_60_pre = band_power(raw, 55, 65)
    raw.notch_filter(freqs=NOTCH_FREQS, verbose="WARNING")
    p_alpha_post = band_power(raw, 8, 13)
    p_60_post = band_power(raw, 55, 65)
    metrics["p_alpha_pre_post_ratio"] = round(p_alpha_post / max(p_alpha_pre, 1e-30), 4)
    metrics["p_60Hz_attenuation"] = round(p_60_pre / max(p_60_post, 1e-30), 1)

    raw_ica = raw.copy().filter(l_freq=1.0, h_freq=100.0, fir_design="firwin", verbose="WARNING")
    raw_analysis = raw.copy().filter(l_freq=1.0, h_freq=20.0, fir_design="firwin", verbose="WARNING")
    raw_ica.set_eeg_reference("average", projection=False, verbose="WARNING")
    raw_analysis.set_eeg_reference("average", projection=False, verbose="WARNING")

    nc = NoisyChannels(raw_ica, random_state=42)
    nc.find_all_bads()
    bads = [str(b) for b in nc.get_bads()]
    metrics["n_bads"] = len(bads)

    raw_ica.info["bads"] = list(bads)
    raw_analysis.info["bads"] = list(bads)
    if bads:
        raw_ica.interpolate_bads(reset_bads=True, verbose="WARNING")
        raw_analysis.interpolate_bads(reset_bads=True, verbose="WARNING")

    if verbose:
        print(f"  [{task}] dur={metrics['movie_duration_s']}s  bads={len(bads)}  "
              f"60Hz atten={metrics['p_60Hz_attenuation']}x", flush=True)

    return raw_ica, raw_analysis, metrics, bads


def prepare_analysis_cached(subject_id, task, bads):
    """Cheap regeneration of raw_analysis using cached bads (no PyPREP search).
    Mirrors the raw_analysis path of prepare_one_movie exactly: load, montage,
    crop, notch, filter 1 to 20, CAR, interpolate cached bads. Returns raw_analysis
    at the native rate (caller resamples)."""
    set_path, events_path = find_eeg_files(subject_id, task)
    if set_path is None:
        return None
    raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose="ERROR")
    raw.set_montage(mne.channels.make_standard_montage("GSN-HydroCel-129"),
                    on_missing="warn", match_case=False)
    v_start, v_stop = parse_movie_window(events_path)
    if v_start is None or v_stop is None:
        v_start, v_stop = 0.0, float(raw.times[-1])
    raw.crop(tmin=v_start, tmax=min(v_stop, float(raw.times[-1])))
    raw.notch_filter(freqs=NOTCH_FREQS, verbose="ERROR")
    raw_analysis = raw.copy().filter(l_freq=1.0, h_freq=20.0, fir_design="firwin", verbose="ERROR")
    raw_analysis.set_eeg_reference("average", projection=False, verbose="ERROR")
    raw_analysis.info["bads"] = list(bads)
    if bads:
        raw_analysis.interpolate_bads(reset_bads=True, verbose="ERROR")
    return raw_analysis


# ------------------------------------------------------------------ fit + cache

def fit_subject(sid, verbose=True):
    """Re-fit ICA for one subject across the available movies and cache it.
    Returns a status dict. Resumable: returns early if cache is complete."""
    if cache_complete(sid):
        return {"sid": sid, "status": "cached"}

    t0 = time.time()
    prepped = []   # (task, raw_ica_200, metrics, bads)
    bads_by_movie = {}
    crop_n_by_movie = {}
    used, failed = [], []
    for task in MOVIES:
        set_path, events_path = find_eeg_files(sid, task)
        if set_path is None:
            failed.append([task, "set_not_found"])
            continue
        try:
            raw_ica, raw_analysis, metrics, bads = prepare_one_movie_capture(
                sid, task, set_path, events_path, verbose=verbose)
            # modification 2: resample raw_ica to 200 Hz before the ICA fit
            if float(raw_ica.info["sfreq"]) > ICA_SFREQ:
                raw_ica.resample(sfreq=ICA_SFREQ, verbose="ERROR")
            dur_s = float(raw_ica.times[-1])
            if dur_s < MIN_DUR_S.get(task, 0.0):
                failed.append([task, f"too_short_{dur_s:.0f}s"])
                continue
            prepped.append((task, raw_ica, metrics, bads))
            bads_by_movie[task] = bads
            crop_n_by_movie[task] = int(raw_ica.n_times)
            used.append(task)
            del raw_analysis
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            failed.append([task, f"prep:{type(e).__name__}:{str(e)[:120]}"])

    if not prepped:
        return {"sid": sid, "status": "no_usable_movies", "failed": failed}

    # concatenate raw_ica (200 Hz) and fit ICA verbatim (rs=42, infomax extended)
    raw_ica_concat = mne.concatenate_raws([ri.copy() for (_, ri, _, _) in prepped],
                                          verbose="ERROR")
    concat_dur = float(raw_ica_concat.times[-1])
    ica = mne.preprocessing.ICA(
        n_components=20, method="infomax", fit_params={"extended": True},
        max_iter=500, random_state=42,
    )
    ica.fit(raw_ica_concat, picks="eeg", verbose="ERROR")

    labels = label_components(raw_ica_concat, ica, method="iclabel")
    probs = [float(p) for p in labels["y_pred_proba"]]
    label_list = list(labels["labels"])
    counts = {lab: int(sum(1 for x in label_list if x == lab)) for lab in
              ["brain", "muscle artifact", "eye blink", "heart beat",
               "line noise", "channel noise", "other"]}
    median_conf = round(float(np.median(probs)), 3)
    reject_idx = [i for i, lab in enumerate(label_list) if lab in REJECT_LABELS]
    eyeblink_idx = [i for i, lab in enumerate(label_list) if lab == "eye blink"]
    rejectB_idx = [i for i in reject_idx if i not in eyeblink_idx]

    # write cache (ICA + json), SSD-robust
    retry(lambda: ica.save(str(ica_path(sid)), overwrite=True, verbose="ERROR"),
          what=f"ica.save {sid}")
    meta = {
        "sid": sid, "complete": True, "ica_sfreq": ICA_SFREQ, "native_sfreq": 500.0,
        "downsample_before_ica": True,
        "movies_used": used, "movies_failed": failed,
        "labels": label_list, "probs": probs, "counts": counts,
        "median_conf": median_conf,
        "reject_idx": reject_idx, "eyeblink_idx": eyeblink_idx,
        "rejectB_idx": rejectB_idx, "n_excluded": len(reject_idx),
        "bads_by_movie": bads_by_movie, "crop_n_by_movie": crop_n_by_movie,
        "concat_dur_s": round(concat_dur, 2),
        "wall_s": round(time.time() - t0, 1),
    }
    retry(lambda: json_path(sid).write_text(json.dumps(meta)), what=f"json {sid}")
    if verbose:
        print(f"  {sid}: fit done in {meta['wall_s']}s, movies={used}, "
              f"eyeblink ICs={eyeblink_idx}, reject ICs={reject_idx}, conf={median_conf}",
              flush=True)
    return {"sid": sid, "status": "fit", **{k: meta[k] for k in
            ("movies_used", "eyeblink_idx", "reject_idx", "wall_s")}}


# ------------------------------------------------------------------ faithfulness

def stored_fif_path(sid, rel, task):
    return STORED_DIR / f"preprocessed_movies_{rel}" / \
        f"{sid}_{task}_preproc_v3concat_raw.fif"


def regenerate_condition_A(sid, task, ica, reject_idx, bads):
    """Regenerate condition A (full removal) at 200 Hz from cached ICA and bads:
    raw_analysis -> resample 200 -> ica.apply(exclude=reject_idx) -> re-CAR."""
    ra = prepare_analysis_cached(sid, task, bads)
    if ra is None:
        return None
    if float(ra.info["sfreq"]) > ICA_SFREQ:
        ra.resample(sfreq=ICA_SFREQ, verbose="ERROR")
    ica2 = ica.copy()
    ica2.exclude = list(reject_idx)
    clean = ica2.apply(ra.copy(), verbose="ERROR")
    clean.set_eeg_reference("average", projection=False, verbose="ERROR")
    return clean


def faithfulness_check(sids, bal):
    print("\n" + "=" * 70)
    print("STEP 2: faithfulness check (regenerated condition A vs stored FIF)")
    print("=" * 70)
    rel_lookup = dict(zip(bal.participant_id, bal.release))
    all_ok = True
    summary = []
    for sid in sids:
        if not cache_complete(sid):
            print(f"  {sid}: NOT CACHED, run fit first. STOP.")
            return False, summary
        with json_path(sid).open() as f:
            meta = json.load(f)
        ica = mne.preprocessing.read_ica(str(ica_path(sid)), verbose="ERROR")
        reject_idx = meta["reject_idx"]
        med_rs = []
        for task in meta["movies_used"]:
            stored_p = stored_fif_path(sid, rel_lookup[sid], task)
            if not stored_p.exists():
                continue
            regen = regenerate_condition_A(sid, task, ica, reject_idx,
                                           meta["bads_by_movie"][task])
            stored = mne.io.read_raw_fif(str(stored_p), preload=True, verbose="ERROR")
            # align channels and length
            common = [c for c in regen.ch_names if c in stored.ch_names]
            a = regen.copy().pick(common).get_data()
            b = stored.copy().pick(common).get_data()
            T = min(a.shape[1], b.shape[1])
            a, b = a[:, :T], b[:, :T]
            rs = []
            for ci in range(a.shape[0]):
                av, bv = a[ci] - a[ci].mean(), b[ci] - b[ci].mean()
                den = np.sqrt((av ** 2).sum() * (bv ** 2).sum())
                rs.append(float((av * bv).sum() / den) if den > 0 else np.nan)
            med_rs.append(float(np.nanmedian(rs)))
        subj_med = float(np.nanmedian(med_rs)) if med_rs else np.nan
        ok = subj_med > 0.95
        all_ok = all_ok and ok
        summary.append({"sid": sid, "median_r": subj_med, "per_movie": med_rs})
        print(f"  {sid}: median per-channel r = {subj_med:.4f} over {len(med_rs)} movies "
              f"-> {'OK' if ok else ('LOW' if subj_med > 0.90 else 'FAIL')}")
    print(f"\n  faithfulness {'PASS' if all_ok else 'CHECK'} "
          f"(all median r > 0.95: {all_ok})")
    if not all_ok and any(s["median_r"] <= 0.90 for s in summary):
        print("  At least one subject below 0.90: regeneration not faithful. STOP.")
    print("STEP 2 COMPLETE")
    return all_ok, summary


# ------------------------------------------------------------------ cli

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fit", "faithfulness"], default="fit")
    ap.add_argument("--sids", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="exit cleanly after this many minutes of fitting (for "
                         "environments that reap long background jobs); resume by "
                         "relaunching, the cache is the checkpoint")
    args = ap.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    bal = build_balanced_150()
    print(f"balanced-150: {len(bal)} subjects, sex={bal.sex.value_counts().to_dict()}, "
          f"cache={CACHE_DIR}", flush=True)

    if args.mode == "faithfulness":
        sids = args.sids if args.sids else list(bal.participant_id.head(args.n))
        faithfulness_check(sids, bal)
        return

    # fit mode
    if args.sids:
        sids = args.sids
    elif args.all:
        sids = list(bal.participant_id)
    else:
        sids = list(bal.participant_id.head(1))  # default: one test subject
    if args.limit:
        sids = sids[:args.limit]

    print(f"FIT mode over {len(sids)} subject(s)"
          + (f", budget {args.max_minutes} min" if args.max_minutes else ""), flush=True)
    done, fitted, failed = 0, 0, 0
    budget_s = args.max_minutes * 60 if args.max_minutes else None
    t0 = time.time()
    for i, sid in enumerate(sids):
        if budget_s is not None and not cache_complete(sid) and (time.time() - t0) > budget_s:
            print(f"TIME BUDGET reached ({args.max_minutes} min) before {sid}; "
                  f"exiting cleanly to resume later", flush=True)
            break
        try:
            res = retry(lambda: fit_subject(sid, verbose=True), tries=2, wait=20,
                        what=f"fit {sid}")
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            res = {"sid": sid, "status": f"error:{type(e).__name__}"}
        st = res.get("status")
        if st == "cached":
            done += 1
        elif st == "fit":
            fitted += 1
        else:
            failed += 1
            print(f"  {sid}: status={st}", flush=True)
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            print(f"PROGRESS {i + 1}/{len(sids)}: cached={done} fitted={fitted} "
                  f"failed={failed}, elapsed={el / 60:.1f}min", flush=True)

    # report completeness over the balanced-150
    n_complete = sum(cache_complete(s) for s in bal.participant_id)
    comp_sids = [s for s in bal.participant_id if cache_complete(s)]
    sexmap = dict(zip(bal.participant_id, bal.sex))
    nM = sum(1 for s in comp_sids if sexmap[s] == "M")
    nF = sum(1 for s in comp_sids if sexmap[s] == "F")
    print(f"\nCACHE COMPLETE for {n_complete}/150 (M={nM} F={nF})", flush=True)
    if n_complete == 150:
        print("ALL 150 ICA FITS CACHED", flush=True)
    print("STEP fit batch done", flush=True)


if __name__ == "__main__":
    main()
