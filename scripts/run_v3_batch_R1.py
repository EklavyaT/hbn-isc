"""Run v3 preprocessing on the full R1 cohort (132 subjects with CBCL labels).

Adds over scripts/run_v3_batch.py:
  - Reads from data/all_R1_to_process.txt
  - Writes FIFs to outputs/preprocessed_R1/
  - Idempotent: skips if FIF for that subject already exists
  - Per-subject QC flag (review if std_p50>30 uV OR conf<0.55 OR ic_brain<5)
  - Writes outputs/R1_results.csv (one row per processed subject) and
    outputs/R1_needs_review.csv (subset flagged for review)
  - Failures appended to logs/R1_failures.log; batch continues on error
  - Progress line every 10 subjects + a heartbeat after every subject
  - Incrementally appends to the CSV after each subject so a crash mid-run
    still leaves usable partial results
"""
import csv
import os
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
SUBJECT_LIST = PROJECT_ROOT / "data" / "all_R1_to_process.txt"
OUT_DIR = PROJECT_ROOT / "outputs" / "preprocessed_R1"
RESULTS_CSV = PROJECT_ROOT / "outputs" / "R1_results.csv"
NEEDS_REVIEW_CSV = PROJECT_ROOT / "outputs" / "R1_needs_review.csv"
FAILURES_LOG = PROJECT_ROOT / "logs" / "R1_failures.log"

REJECT_LABELS = {"eye blink", "muscle artifact", "heart beat", "channel noise", "line noise"}
NOTCH_FREQS = [60.0, 120.0]
TASK = "RestingState"

CSV_FIELDS = [
    "subject", "failed", "error", "skipped",
    "orig_duration_s", "orig_n_channels", "orig_sfreq",
    "n_bads", "bads_by_deviation", "bads_by_hf_noise", "bads_by_correlation",
    "bads_by_ransac", "bads_by_SNR", "bads_by_dropout",
    "ic_brain", "ic_muscle", "ic_eyeblink", "ic_heart", "ic_line",
    "ic_channel", "ic_other",
    "icalabel_median_conf",
    "n_ICs_excluded", "excluded_labels",
    "std_p10_uV", "std_p50_uV", "std_p90_uV",
    "any_nan", "any_inf",
    "wall_clock_sec",
    "qc_flag", "qc_reason",
]


def find_set_path(subject_id):
    """RS .set may live under sub-NDARxxx (BIDS) or sub_NDARxxx (legacy local)."""
    for stem in (subject_id, subject_id.replace("-", "_")):
        candidate = PROJECT_ROOT / "data" / stem / "eeg" / f"{subject_id}_task-{TASK}_eeg.set"
        if candidate.exists():
            return candidate
    return None


def find_existing_fif(subject_id):
    """Check both naming conventions for the output FIF."""
    for stem in (subject_id, subject_id.replace("-", "_")):
        candidate = OUT_DIR / f"{stem}_RestingState_preproc_v3_raw.fif"
        if candidate.exists():
            return candidate
    return None


def compute_qc(metrics):
    reasons = []
    if metrics.get("std_p50_uV") is not None and metrics["std_p50_uV"] > 30:
        reasons.append(f"std_p50={metrics['std_p50_uV']:.1f}uV>30")
    if metrics.get("icalabel_median_conf") is not None and metrics["icalabel_median_conf"] < 0.55:
        reasons.append(f"conf={metrics['icalabel_median_conf']:.2f}<0.55")
    if metrics.get("ic_brain") is not None and metrics["ic_brain"] < 5:
        reasons.append(f"ic_brain={metrics['ic_brain']}<5")
    metrics["qc_flag"] = "review" if reasons else "pass"
    metrics["qc_reason"] = ";".join(reasons)
    return metrics


def process_subject(subject_id, eeg_path, out_path):
    t0 = time.time()
    metrics = {"subject": subject_id, "failed": False, "error": "", "skipped": False}

    raw = mne.io.read_raw_eeglab(str(eeg_path), preload=True, verbose="WARNING")
    metrics["orig_duration_s"] = round(float(raw.times[-1]), 2)
    metrics["orig_n_channels"] = len(raw.ch_names)
    metrics["orig_sfreq"] = float(raw.info["sfreq"])

    raw.set_montage(
        mne.channels.make_standard_montage("GSN-HydroCel-129"),
        on_missing="warn",
        match_case=False,
    )
    raw.notch_filter(freqs=NOTCH_FREQS, verbose="WARNING")

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

    raw_ica.info["bads"] = bads
    raw_analysis.info["bads"] = list(bads)
    if bads:
        raw_ica.interpolate_bads(reset_bads=True, verbose="WARNING")
        raw_analysis.interpolate_bads(reset_bads=True, verbose="WARNING")

    ica = mne.preprocessing.ICA(
        n_components=20, method="infomax", fit_params={"extended": True},
        max_iter=500, random_state=42,
    )
    ica.fit(raw_ica, picks="eeg", verbose="WARNING")

    labels = label_components(raw_ica, ica, method="iclabel")
    probs = [float(p) for p in labels["y_pred_proba"]]
    label_list = list(labels["labels"])
    counts = {lab: int(sum(1 for x in label_list if x == lab)) for lab in
              ["brain", "muscle artifact", "eye blink", "heart beat",
               "line noise", "channel noise", "other"]}
    metrics["ic_brain"] = counts["brain"]
    metrics["ic_muscle"] = counts["muscle artifact"]
    metrics["ic_eyeblink"] = counts["eye blink"]
    metrics["ic_heart"] = counts["heart beat"]
    metrics["ic_line"] = counts["line noise"]
    metrics["ic_channel"] = counts["channel noise"]
    metrics["ic_other"] = counts["other"]
    metrics["icalabel_median_conf"] = round(float(np.median(probs)), 3)

    bad_ic_idx = [i for i, lab in enumerate(label_list) if lab in REJECT_LABELS]
    metrics["n_ICs_excluded"] = len(bad_ic_idx)
    metrics["excluded_labels"] = ";".join(label_list[i] for i in bad_ic_idx)

    ica.exclude = bad_ic_idx
    raw_clean = ica.apply(raw_analysis.copy(), verbose="WARNING")
    raw_clean.set_eeg_reference("average", projection=False, verbose="WARNING")
    raw_clean.resample(sfreq=200.0, verbose="WARNING")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_clean.save(str(out_path), overwrite=True, verbose="WARNING")

    data = raw_clean.get_data()
    stds_uv = data.std(axis=1) * 1e6
    metrics["std_p10_uV"] = round(float(np.percentile(stds_uv, 10)), 2)
    metrics["std_p50_uV"] = round(float(np.percentile(stds_uv, 50)), 2)
    metrics["std_p90_uV"] = round(float(np.percentile(stds_uv, 90)), 2)
    metrics["any_nan"] = bool(np.any(np.isnan(data)))
    metrics["any_inf"] = bool(np.any(np.isinf(data)))
    metrics["wall_clock_sec"] = round(time.time() - t0, 1)

    return compute_qc(metrics)


def append_csv_row(row, fieldnames, path):
    write_header = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


def log_failure(subject_id, exc):
    FAILURES_LOG.parent.mkdir(parents=True, exist_ok=True)
    with FAILURES_LOG.open("a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {subject_id}\n")
        f.write(f"{'='*60}\n")
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=f)


def main():
    if not SUBJECT_LIST.exists():
        sys.exit(f"Subject list not found: {SUBJECT_LIST}")
    subjects = [s.strip() for s in SUBJECT_LIST.read_text().splitlines() if s.strip()]
    print(f"=== R1 batch: {len(subjects)} subject(s) ===", flush=True)

    # Track which subjects already have rows so a restart does not duplicate them.
    done_set = set()
    if RESULTS_CSV.exists():
        prev = pd.read_csv(RESULTS_CSV)
        done_set = set(prev["subject"].tolist())
        print(f"Resuming: {len(done_set)} subject(s) already have CSV rows", flush=True)

    t_batch_start = time.time()
    n_proc = n_skip = n_fail = 0

    for i, sid in enumerate(subjects, 1):
        if sid in done_set:
            n_skip += 1
            continue

        existing_fif = find_existing_fif(sid)
        if existing_fif is not None:
            print(f"[{i}/{len(subjects)}] SKIP {sid} (FIF exists at {existing_fif.name})", flush=True)
            row = {"subject": sid, "failed": False, "skipped": True, "qc_flag": "skipped"}
            append_csv_row(row, CSV_FIELDS, RESULTS_CSV)
            n_skip += 1
            continue

        eeg_path = find_set_path(sid)
        if eeg_path is None:
            print(f"[{i}/{len(subjects)}] FAIL {sid}: RestingState .set not found", flush=True)
            row = {"subject": sid, "failed": True, "error": "set_not_found"}
            row = compute_qc(row)
            append_csv_row(row, CSV_FIELDS, RESULTS_CSV)
            n_fail += 1
            continue

        out_path = OUT_DIR / f"{sid}_RestingState_preproc_v3_raw.fif"
        try:
            metrics = process_subject(sid, eeg_path, out_path)
            append_csv_row(metrics, CSV_FIELDS, RESULTS_CSV)
            n_proc += 1
            flag = metrics["qc_flag"]
            print(f"[{i}/{len(subjects)}] OK {sid} {metrics['wall_clock_sec']}s | "
                  f"bads={metrics['n_bads']} ICs_excl={metrics['n_ICs_excluded']} "
                  f"brain={metrics['ic_brain']} std50={metrics['std_p50_uV']}uV | {flag}",
                  flush=True)
        except Exception as e:
            log_failure(sid, e)
            row = {"subject": sid, "failed": True, "error": str(e)[:300]}
            row = compute_qc(row)
            append_csv_row(row, CSV_FIELDS, RESULTS_CSV)
            n_fail += 1
            print(f"[{i}/{len(subjects)}] FAIL {sid}: {type(e).__name__}: {str(e)[:120]}", flush=True)

        if i % 10 == 0:
            elapsed = (time.time() - t_batch_start) / 60
            done_count = n_proc + n_skip + n_fail
            rate = done_count / max(elapsed, 1e-3)
            remaining = (len(subjects) - i) / max(rate, 1e-3)
            print(f"  ... progress: {i}/{len(subjects)} | proc={n_proc} skip={n_skip} fail={n_fail} | "
                  f"elapsed={elapsed:.1f}m | est remaining={remaining:.1f}m", flush=True)

    # Build needs-review CSV from the full results.
    if RESULTS_CSV.exists():
        df = pd.read_csv(RESULTS_CSV)
        review = df[df["qc_flag"] == "review"]
        review.to_csv(NEEDS_REVIEW_CSV, index=False)
        print(f"\nWrote {RESULTS_CSV} ({len(df)} rows) and {NEEDS_REVIEW_CSV} ({len(review)} rows)", flush=True)

    total_min = (time.time() - t_batch_start) / 60
    print(f"\n=== DONE ===", flush=True)
    print(f"  processed: {n_proc}", flush=True)
    print(f"  skipped: {n_skip}", flush=True)
    print(f"  failed: {n_fail}", flush=True)
    print(f"  total wall clock: {total_min:.1f} min", flush=True)


if __name__ == "__main__":
    main()
