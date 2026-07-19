"""Cohort batch runner for the v3 concatenated-ICA movie pipeline.

Parameterized by --release {R1, R2, R3, R4, ...}. Defaults to R1 for back-compat.
Reads data/{REL}_subjects_full_movies.txt.
For each subject:
  - Skip if all 4 expected v3concat FIFs already exist (idempotent).
  - Run scripts/preprocess_movies_v3_concat.process_subject().
  - Append per-(subject, movie) rows to outputs/{REL}_movies_results.csv.
  - Continue on failure; failures appended to logs/{REL}_movies_failures.log.

Also writes outputs/{REL}_movies_summary.csv: one row per subject with the
worst-case across that subject's 4 movies (max n_bads, min ic_brain,
min icalabel_median_conf, max std_p50_uV, n_movies_pass, qc_flag).

FIFs are written to outputs/preprocessed_movies_{REL}/.

QC thresholds (looser than RestingState bars to account for shorter clips):
  std_p50_uV > 40, icalabel_median_conf < 0.50, ic_brain < 3.

Progress line every 10 subjects. Heartbeat after every subject.
"""
import argparse
import csv
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

# Reuse the per-subject pipeline.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess_movies_v3_concat import (  # type: ignore
    PER_MOVIE_FIELDS, DEFAULT_TASKS,
    process_subject, append_csv_row, compute_qc,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# These get rebound in main() once we know the release.
RELEASE = "R1"
SUBJECT_LIST = PROJECT_ROOT / "data" / f"{RELEASE}_subjects_full_movies.txt"
RESULTS_CSV = PROJECT_ROOT / "outputs" / f"{RELEASE}_movies_results.csv"
SUMMARY_CSV = PROJECT_ROOT / "outputs" / f"{RELEASE}_movies_summary.csv"
FAILURES_LOG = PROJECT_ROOT / "logs" / f"{RELEASE}_movies_failures.log"
OUT_DIR = PROJECT_ROOT / "outputs" / f"preprocessed_movies_{RELEASE}"

SUMMARY_FIELDS = [
    "participant_id",
    "n_movies_processed", "n_movies_failed", "n_movies_pass", "n_movies_review",
    "max_n_bads", "min_ic_brain", "min_icalabel_median_conf", "max_std_p50_uV",
    "subject_qc_flag", "subject_qc_reason",
]


def expected_fifs(subject_id):
    return [OUT_DIR / f"{subject_id}_{task}_preproc_v3concat_raw.fif"
            for task in DEFAULT_TASKS]


def all_fifs_exist(subject_id):
    return all(p.exists() for p in expected_fifs(subject_id))


def log_failure(subject_id, exc):
    FAILURES_LOG.parent.mkdir(parents=True, exist_ok=True)
    with FAILURES_LOG.open("a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {subject_id}\n")
        f.write(f"{'='*60}\n")
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=f)


def summarize_subject(subject_id, rows):
    """rows: list of per-movie metric dicts for this subject. Returns one summary dict."""
    s = {"participant_id": subject_id}
    s["n_movies_processed"] = len(rows)
    s["n_movies_failed"] = sum(1 for r in rows if r.get("failed"))
    s["n_movies_pass"] = sum(1 for r in rows if r.get("qc_flag") == "pass")
    s["n_movies_review"] = sum(1 for r in rows if r.get("qc_flag") == "review")

    successful = [r for r in rows if not r.get("failed")]
    if successful:
        s["max_n_bads"] = max((r.get("n_bads") or 0) for r in successful)
        s["min_ic_brain"] = min((r.get("ic_brain") if r.get("ic_brain") is not None else 999)
                                 for r in successful)
        s["min_icalabel_median_conf"] = min(
            (r.get("icalabel_median_conf") if r.get("icalabel_median_conf") is not None else 1.0)
            for r in successful
        )
        s["max_std_p50_uV"] = max(
            (r.get("std_p50_uV") if r.get("std_p50_uV") is not None else 0.0)
            for r in successful
        )
    else:
        s["max_n_bads"] = ""
        s["min_ic_brain"] = ""
        s["min_icalabel_median_conf"] = ""
        s["max_std_p50_uV"] = ""

    if s["n_movies_failed"] > 0:
        s["subject_qc_flag"] = "fail"
        s["subject_qc_reason"] = f"{s['n_movies_failed']} of {s['n_movies_processed']} movies failed"
    elif s["n_movies_review"] > 0:
        s["subject_qc_flag"] = "review"
        s["subject_qc_reason"] = f"{s['n_movies_review']} movie(s) flagged review"
    else:
        s["subject_qc_flag"] = "pass"
        s["subject_qc_reason"] = ""
    return s


def main():
    global RELEASE, SUBJECT_LIST, RESULTS_CSV, SUMMARY_CSV, FAILURES_LOG, OUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", default="R1",
                        help="HBN release tag (R1, R2, R3, R4, ...)")
    args = parser.parse_args()
    RELEASE = args.release
    SUBJECT_LIST = PROJECT_ROOT / "data" / f"{RELEASE}_subjects_full_movies.txt"
    RESULTS_CSV = PROJECT_ROOT / "outputs" / f"{RELEASE}_movies_results.csv"
    SUMMARY_CSV = PROJECT_ROOT / "outputs" / f"{RELEASE}_movies_summary.csv"
    FAILURES_LOG = PROJECT_ROOT / "logs" / f"{RELEASE}_movies_failures.log"
    OUT_DIR = PROJECT_ROOT / "outputs" / f"preprocessed_movies_{RELEASE}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not SUBJECT_LIST.exists():
        sys.exit(f"Subject list not found: {SUBJECT_LIST}")
    subjects = [s.strip() for s in SUBJECT_LIST.read_text().splitlines() if s.strip()]
    print(f"=== {RELEASE} movies v3concat batch: {len(subjects)} subject(s) ===", flush=True)
    print(f"  OUT_DIR={OUT_DIR}", flush=True)

    # Track which subjects already have rows so a restart does not duplicate them.
    done_set = set()
    if RESULTS_CSV.exists():
        prev = pd.read_csv(RESULTS_CSV)
        done_set = set(prev["participant_id"].tolist())
        print(f"Resuming: {len(done_set)} subject(s) already have CSV rows", flush=True)

    t_batch_start = time.time()
    n_proc = n_skip = n_fail = 0

    for i, sid in enumerate(subjects, 1):
        if sid in done_set and all_fifs_exist(sid):
            print(f"[{i}/{len(subjects)}] SKIP {sid} (already in CSV and FIFs exist)", flush=True)
            n_skip += 1
            continue

        if all_fifs_exist(sid) and sid not in done_set:
            print(f"[{i}/{len(subjects)}] {sid}: FIFs exist but no CSV row, will reprocess", flush=True)

        try:
            rows = process_subject(sid, DEFAULT_TASKS, OUT_DIR, verbose=False)
            for r in rows:
                append_csv_row(r, PER_MOVIE_FIELDS, RESULTS_CSV)
            summary = summarize_subject(sid, rows)
            n_proc += 1
            print(f"[{i}/{len(subjects)}] OK {sid} | "
                  f"pass={summary['n_movies_pass']} review={summary['n_movies_review']} "
                  f"fail={summary['n_movies_failed']} | "
                  f"max_bads={summary['max_n_bads']} min_brain={summary['min_ic_brain']} "
                  f"min_conf={summary['min_icalabel_median_conf']} "
                  f"max_std50={summary['max_std_p50_uV']} | "
                  f"flag={summary['subject_qc_flag']}", flush=True)
        except Exception as e:
            log_failure(sid, e)
            # Write 4 failed rows for this subject so the CSV stays one-per-(sub, movie).
            for task in DEFAULT_TASKS:
                row = {"participant_id": sid, "movie": task, "failed": True,
                       "error": f"subject:{type(e).__name__}:{str(e)[:200]}"}
                row = compute_qc(row)
                append_csv_row(row, PER_MOVIE_FIELDS, RESULTS_CSV)
            n_fail += 1
            print(f"[{i}/{len(subjects)}] FAIL {sid}: {type(e).__name__}: {str(e)[:120]}",
                  flush=True)

        if i % 10 == 0:
            elapsed = (time.time() - t_batch_start) / 60
            done = n_proc + n_skip + n_fail
            rate = done / max(elapsed, 1e-3)
            remaining = (len(subjects) - i) / max(rate, 1e-3)
            print(f"  ... progress: {i}/{len(subjects)} | proc={n_proc} skip={n_skip} fail={n_fail} | "
                  f"elapsed={elapsed:.1f}m | est remaining={remaining:.1f}m", flush=True)

    # Build per-subject summary CSV from the per-movie results.
    if RESULTS_CSV.exists():
        df = pd.read_csv(RESULTS_CSV)
        summary_rows = []
        for sid, group in df.groupby("participant_id"):
            rows = group.to_dict("records")
            for r in rows:
                # csv-loaded NaN -> None, so normalize qc_flag
                if pd.isna(r.get("qc_flag")):
                    r["qc_flag"] = ""
            summary_rows.append(summarize_subject(sid, rows))
        SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
        with SUMMARY_CSV.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
            w.writeheader()
            for r in summary_rows:
                w.writerow({k: r.get(k, "") for k in SUMMARY_FIELDS})
        print(f"\nWrote {RESULTS_CSV} ({len(df)} rows) and {SUMMARY_CSV} ({len(summary_rows)} rows)",
              flush=True)

    total_min = (time.time() - t_batch_start) / 60
    print("\n=== DONE ===", flush=True)
    print(f"  processed: {n_proc}", flush=True)
    print(f"  skipped: {n_skip}", flush=True)
    print(f"  failed: {n_fail}", flush=True)
    print(f"  total wall clock: {total_min:.1f} min", flush=True)


if __name__ == "__main__":
    main()
