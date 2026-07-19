"""v3 movie pipeline with ICA fit on the concatenation of all 4 movies.

Identical to scripts/preprocess_movie_v3.py except for the ICA stage:
  - Per-movie pre-ICA: load -> set montage -> crop to [video_start, video_stop]
    -> notch -> two filter copies -> CAR -> PyPREP NoisyChannels on the per-
    movie raw_ica (each movie is a separate recording session structurally)
    -> interpolate bads on both copies.
  - Concatenate the 4 raw_ica streams (mne.concatenate_raws inserts a BAD
    boundary annotation between movies; ICA / filter / CAR all skip them).
  - Fit ICA on the concatenated raw_ica.
  - Run ICALabel once on the concatenated stream + ICA.
  - For each movie's raw_analysis: ica.apply -> re-CAR -> resample 200 Hz -> save.

DiaryOfAWimpyKid alone is ~117 s; concatenating gives ICA ~10-15 min of data
across 4 movies, which should stabilize the decomposition for short clips.

Output: outputs/preprocessed_movies_R1/{subject}_{task}_preproc_v3concat_raw.fif
Metrics CSV: outputs/{subject}_movies_v3concat_results.csv (one row per movie)
"""
import argparse
import csv
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import mne
from mne_icalabel import label_components
from pyprep.find_noisy_channels import NoisyChannels

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "preprocessed_movies_R1"

REJECT_LABELS = {"eye blink", "muscle artifact", "heart beat", "channel noise", "line noise"}
NOTCH_FREQS = [60.0, 120.0]
DEFAULT_TASKS = ["DespicableMe", "DiaryOfAWimpyKid", "FunwithFractals", "ThePresent"]

PER_MOVIE_FIELDS = [
    "participant_id", "movie", "failed", "error",
    "orig_duration_s", "movie_duration_s", "video_start_s", "video_stop_s",
    "orig_n_channels", "orig_sfreq",
    "n_bads", "bads_by_deviation", "bads_by_hf_noise", "bads_by_correlation",
    "bads_by_ransac", "bads_by_SNR", "bads_by_dropout",
    # Concat-level (same value across all 4 rows for a given subject):
    "concat_dur_s",
    "ic_brain", "ic_muscle", "ic_eyeblink", "ic_heart", "ic_line",
    "ic_channel", "ic_other",
    "icalabel_median_conf",
    "n_ICs_excluded", "excluded_labels",
    # Per-movie post-ICA:
    "std_p10_uV", "std_p50_uV", "std_p90_uV",
    "any_nan", "any_inf",
    "p_alpha_pre_post_ratio", "p_60Hz_attenuation",
    "wall_clock_sec",
    "qc_flag", "qc_reason",
]


def find_eeg_files(subject_id, task):
    for stem in (subject_id, subject_id.replace("-", "_")):
        eeg_dir = PROJECT_ROOT / "data" / stem / "eeg"
        set_path = eeg_dir / f"{subject_id}_task-{task}_eeg.set"
        events_path = eeg_dir / f"{subject_id}_task-{task}_events.tsv"
        if set_path.exists():
            return set_path, events_path if events_path.exists() else None
    return None, None


def parse_movie_window(events_path):
    if events_path is None:
        return None, None
    df = pd.read_csv(events_path, sep="\t")
    starts = df.loc[df["value"].astype(str) == "video_start", "onset"].astype(float).tolist()
    stops = df.loc[df["value"].astype(str) == "video_stop", "onset"].astype(float).tolist()
    if not starts or not stops:
        return None, None
    return float(starts[0]), float(stops[0])


def band_power(raw, lo, hi, fmax=None):
    fmax = fmax if fmax is not None else hi + 5
    psd = raw.compute_psd(fmin=max(0.5, lo - 5), fmax=fmax, n_fft=2048, verbose="WARNING")
    freqs = psd.freqs
    mask = (freqs >= lo) & (freqs <= hi)
    return float(psd.get_data().mean(axis=0)[mask].mean())


def compute_qc(metrics, *, std_max=40, conf_min=0.50, brain_min=3):
    reasons = []
    if metrics.get("std_p50_uV") is not None and metrics["std_p50_uV"] > std_max:
        reasons.append(f"std_p50={metrics['std_p50_uV']:.1f}uV>{std_max}")
    if metrics.get("icalabel_median_conf") is not None and metrics["icalabel_median_conf"] < conf_min:
        reasons.append(f"conf={metrics['icalabel_median_conf']:.2f}<{conf_min}")
    if metrics.get("ic_brain") is not None and metrics["ic_brain"] < brain_min:
        reasons.append(f"ic_brain={metrics['ic_brain']}<{brain_min}")
    metrics["qc_flag"] = "review" if reasons else "pass"
    metrics["qc_reason"] = ";".join(reasons)
    return metrics


def append_csv_row(row, fieldnames, path):
    write_header = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


def prepare_one_movie(subject_id, task, set_path, events_path, verbose=True):
    """Returns (raw_ica, raw_analysis, per_movie_metrics) for one movie."""
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
    metrics["bads_by_deviation"] = len(getattr(nc, "bad_by_deviation", []) or [])
    metrics["bads_by_hf_noise"] = len(getattr(nc, "bad_by_hf_noise", []) or [])
    metrics["bads_by_correlation"] = len(getattr(nc, "bad_by_correlation", []) or [])
    metrics["bads_by_ransac"] = len(getattr(nc, "bad_by_ransac", []) or [])
    metrics["bads_by_SNR"] = len(getattr(nc, "bad_by_SNR", []) or [])
    metrics["bads_by_dropout"] = len(getattr(nc, "bad_by_dropout", []) or [])

    raw_ica.info["bads"] = list(bads)
    raw_analysis.info["bads"] = list(bads)
    if bads:
        raw_ica.interpolate_bads(reset_bads=True, verbose="WARNING")
        raw_analysis.interpolate_bads(reset_bads=True, verbose="WARNING")

    if verbose:
        print(f"  [{task}] dur={metrics['movie_duration_s']}s  bads={len(bads)}  "
              f"60Hz atten={metrics['p_60Hz_attenuation']}x  "
              f"alpha post/pre={metrics['p_alpha_pre_post_ratio']}", flush=True)

    return raw_ica, raw_analysis, metrics


def process_subject(subject_id, tasks, out_dir, verbose=True):
    """Run the full concat pipeline for one subject. Returns list of per-movie metric dicts."""
    t_start = time.time()

    # Phase 1: per-movie prep.
    prepped = []  # list of (task, raw_ica, raw_analysis, metrics)
    if verbose:
        print(f"\n=== {subject_id} ===", flush=True)
        print("Phase 1: per-movie load/crop/notch/CAR/PyPREP", flush=True)
    for task in tasks:
        set_path, events_path = find_eeg_files(subject_id, task)
        if set_path is None:
            metrics = {"participant_id": subject_id, "movie": task, "failed": True,
                       "error": "set_not_found"}
            metrics = compute_qc(metrics)
            prepped.append((task, None, None, metrics))
            if verbose:
                print(f"  [{task}] SKIP: .set not found", flush=True)
            continue
        try:
            raw_ica, raw_analysis, metrics = prepare_one_movie(
                subject_id, task, set_path, events_path, verbose=verbose
            )
            prepped.append((task, raw_ica, raw_analysis, metrics))
        except Exception as e:
            traceback.print_exc()
            metrics = {"participant_id": subject_id, "movie": task, "failed": True,
                       "error": f"prep:{type(e).__name__}:{str(e)[:200]}"}
            metrics = compute_qc(metrics)
            prepped.append((task, None, None, metrics))

    usable = [(t, ri, ra, m) for (t, ri, ra, m) in prepped if ri is not None]
    if not usable:
        return [m for (_, _, _, m) in prepped]

    # Phase 2: concatenate raw_ica streams and fit ICA.
    if verbose:
        print(f"\nPhase 2: concatenate {len(usable)} streams and fit ICA", flush=True)
    raw_ica_concat = mne.concatenate_raws(
        [ri.copy() for (_, ri, _, _) in usable], verbose="WARNING"
    )
    concat_dur = float(raw_ica_concat.times[-1])
    if verbose:
        print(f"  concatenated raw_ica: dur={concat_dur:.1f}s, "
              f"n_samples={raw_ica_concat.n_times}, n_ch={len(raw_ica_concat.ch_names)}", flush=True)

    ica = mne.preprocessing.ICA(
        n_components=20, method="infomax", fit_params={"extended": True},
        max_iter=500, random_state=42,
    )
    ica.fit(raw_ica_concat, picks="eeg", verbose="WARNING")

    labels = label_components(raw_ica_concat, ica, method="iclabel")
    probs = [float(p) for p in labels["y_pred_proba"]]
    label_list = list(labels["labels"])
    counts = {lab: int(sum(1 for x in label_list if x == lab)) for lab in
              ["brain", "muscle artifact", "eye blink", "heart beat",
               "line noise", "channel noise", "other"]}
    median_conf = round(float(np.median(probs)), 3)
    bad_ic_idx = [i for i, lab in enumerate(label_list) if lab in REJECT_LABELS]
    excluded_labels_str = ";".join(label_list[i] for i in bad_ic_idx)
    if verbose:
        print(f"  ICA labels: {counts}, median conf={median_conf}", flush=True)
        print(f"  excluding {len(bad_ic_idx)} IC(s): {excluded_labels_str}", flush=True)

    ica.exclude = list(bad_ic_idx)

    # Phase 3: per-movie ICA apply, re-CAR, resample, save.
    if verbose:
        print(f"\nPhase 3: per-movie ICA apply / re-CAR / resample / save", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    completed_metrics = []
    for (task, raw_ica, raw_analysis, metrics) in prepped:
        if raw_ica is None:
            completed_metrics.append(metrics)
            continue
        try:
            metrics["concat_dur_s"] = round(concat_dur, 2)
            metrics["ic_brain"] = counts["brain"]
            metrics["ic_muscle"] = counts["muscle artifact"]
            metrics["ic_eyeblink"] = counts["eye blink"]
            metrics["ic_heart"] = counts["heart beat"]
            metrics["ic_line"] = counts["line noise"]
            metrics["ic_channel"] = counts["channel noise"]
            metrics["ic_other"] = counts["other"]
            metrics["icalabel_median_conf"] = median_conf
            metrics["n_ICs_excluded"] = len(bad_ic_idx)
            metrics["excluded_labels"] = excluded_labels_str

            raw_clean = ica.apply(raw_analysis.copy(), verbose="WARNING")
            raw_clean.set_eeg_reference("average", projection=False, verbose="WARNING")
            raw_clean.resample(sfreq=200.0, verbose="WARNING")

            out_path = out_dir / f"{subject_id}_{task}_preproc_v3concat_raw.fif"
            raw_clean.save(str(out_path), overwrite=True, verbose="WARNING")

            data = raw_clean.get_data()
            stds_uv = data.std(axis=1) * 1e6
            metrics["std_p10_uV"] = round(float(np.percentile(stds_uv, 10)), 2)
            metrics["std_p50_uV"] = round(float(np.percentile(stds_uv, 50)), 2)
            metrics["std_p90_uV"] = round(float(np.percentile(stds_uv, 90)), 2)
            metrics["any_nan"] = bool(np.any(np.isnan(data)))
            metrics["any_inf"] = bool(np.any(np.isinf(data)))
            compute_qc(metrics)
            if verbose:
                print(f"  [{task}] std_p50={metrics['std_p50_uV']}uV  "
                      f"flag={metrics['qc_flag']}", flush=True)
        except Exception as e:
            traceback.print_exc()
            metrics["failed"] = True
            metrics["error"] = f"apply:{type(e).__name__}:{str(e)[:200]}"
            compute_qc(metrics)
        finally:
            completed_metrics.append(metrics)

    wall_clock = round(time.time() - t_start, 1)
    for m in completed_metrics:
        m["wall_clock_sec"] = wall_clock

    if verbose:
        print(f"\n=== {subject_id} done in {wall_clock}s ===", flush=True)

    return completed_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="sub-NDARAC904DMU")
    ap.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    ap.add_argument("--results-csv", default=None)
    args = ap.parse_args()

    results_csv = (Path(args.results_csv) if args.results_csv
                   else PROJECT_ROOT / "outputs" / f"{args.subject}_movies_v3concat_results.csv")
    if results_csv.exists():
        results_csv.unlink()

    metrics_list = process_subject(args.subject, args.tasks, OUT_DIR, verbose=True)
    for m in metrics_list:
        append_csv_row(m, PER_MOVIE_FIELDS, results_csv)

    print("\n| Movie               | dur_s | n_bads | n_ICs_excl | std_p50_uV | flag    |")
    print("|---------------------|-------|--------|------------|------------|---------|")
    for m in metrics_list:
        if m.get("failed"):
            print(f"| {m['movie']:<19} | FAILED: {m.get('error','')[:60]}")
            continue
        print(f"| {m['movie']:<19} | {m.get('movie_duration_s',0):>5.1f} | "
              f"{m.get('n_bads','?'):>6} | {m.get('n_ICs_excluded','?'):>10} | "
              f"{m.get('std_p50_uV','?'):>10} | {m.get('qc_flag','?'):<7} |")
    print(f"\nWrote {results_csv}")


if __name__ == "__main__":
    main()
