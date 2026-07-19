"""Build the per-release subject lists for the movie branch.

Reconstructs two families of list that the movie pipeline needs but that no
script previously generated. Both were verified to reproduce their on-disk
originals exactly, by count and by membership, for all eight releases.

Stage 1, the download driver: data/{REL}_subjects_with_labels.txt
    Rule: participants in release REL with complete CBCL bifactor scores
    (p_factor, attention, internalizing, externalizing all non-null).
    Source: data/{REL}_participants.tsv, fetched by download_hbn_data.sh.
    Counts: 132, 147, 180, 318, 323, 131, 379, 257.

Stage 2, the preprocessing input: data/{REL}_subjects_full_movies.txt
    Rule: subjects that have all four movie .set files present on disk after
    download, intersected with that release's participants.tsv membership.
    This is a file-presence rule, not a QC rule. It defines the 1535 subject
    pre-QC set that is submitted to preprocessing, which the later QC filter in
    build_isc_cohort.py narrows to the 1143 subject analytic cohort.
    Counts: 120, 120, 157, 293, 282, 102, 247, 214, summing to 1535.

Neither stage is circular. Stage 1 reads only the phenotype file. Stage 2 reads
only the downloaded BIDS tree and the phenotype file. Neither reads the lists it
reconstructs, and neither reads any pipeline output such as the master table
(which would be circular, since the master is built from preprocessing results
that require these lists as input).

Important caveat on stage 2. Because the rule enumerates what is actually on
disk, it reproduces the published 1535 only when the download covered the same
subject set the original run used, that is, the stage 1 lists. Downloading a
broader set could enumerate additional subjects and change the cohort. The eight
`_subjects_full_movies.txt` files are therefore also shipped in this repository,
and those shipped copies are authoritative for exact reproduction.

Usage:
    python scripts/build_movie_subject_lists.py --stage labels --verify
    python scripts/build_movie_subject_lists.py --stage full-movies --verify
    python scripts/build_movie_subject_lists.py --verify            # both stages
    python scripts/build_movie_subject_lists.py --data-dir /path/to/data \\
        --out-dir /tmp/check --verify
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RELEASES = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"]
CBCL = ["p_factor", "attention", "internalizing", "externalizing"]
MOVIES = ["DespicableMe", "DiaryOfAWimpyKid", "FunwithFractals", "ThePresent"]

EXPECTED_LABELS = {"R1": 132, "R2": 147, "R3": 180, "R4": 318,
                   "R5": 323, "R6": 131, "R7": 379, "R8": 257}
EXPECTED_FULL = {"R1": 120, "R2": 120, "R3": 157, "R4": 293,
                 "R5": 282, "R6": 102, "R7": 247, "R8": 214}


def participants(data_dir, rel):
    path = data_dir / f"{rel}_participants.tsv"
    if not path.exists():
        sys.exit(
            f"Phenotype file not found: {path}\n"
            "Fetch it with:\n"
            f"  bash scripts/download_hbn_data.sh --releases {rel}"
        )
    return pd.read_csv(path, sep="\t")


def build_labels(data_dir, rel):
    """Stage 1: participants with complete CBCL bifactor scores."""
    p = participants(data_dir, rel)
    missing = [c for c in CBCL if c not in p.columns]
    if missing:
        sys.exit(f"{rel}_participants.tsv is missing columns: {missing}")
    keep = p[p[CBCL].notna().all(axis=1)]
    return sorted(keep["participant_id"].astype(str).tolist())


def subjects_with_all_movies(data_dir):
    """Enumerate the downloaded BIDS tree once, returning subjects that have all
    four movie .set files present."""
    present = set()
    for sub_dir in data_dir.glob("sub-*"):
        eeg = sub_dir / "eeg"
        if not eeg.is_dir():
            continue
        try:
            names = os.listdir(eeg)
        except OSError:
            continue
        if all(any(f"task-{mv}" in n and n.endswith(".set") for n in names)
               for mv in MOVIES):
            present.add(sub_dir.name)
    return present


def build_full_movies(data_dir, rel, present):
    """Stage 2: on-disk movie presence, restricted to this release."""
    p = participants(data_dir, rel)
    in_release = set(p["participant_id"].astype(str))
    return sorted(in_release & present)


def compare(rel, rebuilt, original_path):
    """Diff a rebuilt list against its on-disk original."""
    if not original_path.exists():
        print(f"  {rel}: no original at {original_path.name}, cannot verify")
        return None
    original = [s.strip() for s in original_path.read_text().splitlines() if s.strip()]
    same = set(rebuilt) == set(original)
    extra = sorted(set(rebuilt) - set(original))
    missing = sorted(set(original) - set(rebuilt))
    status = "MATCH" if same and len(rebuilt) == len(original) else "MISMATCH"
    print(f"  {rel}: rebuilt={len(rebuilt):3d} original={len(original):3d} "
          f"count={'ok' if len(rebuilt)==len(original) else 'DIFF'} "
          f"membership={'ok' if same else 'DIFF'}  {status}")
    for s in extra[:5]:
        print(f"       only in rebuilt : {s}")
    for s in missing[:5]:
        print(f"       only in original: {s}")
    return same and len(rebuilt) == len(original)


def run_stage(stage, data_dir, out_dir, verify):
    suffix = {"labels": "subjects_with_labels", "full-movies": "subjects_full_movies"}[stage]
    expected = {"labels": EXPECTED_LABELS, "full-movies": EXPECTED_FULL}[stage]
    print(f"=== stage: {stage} ({suffix}.txt) ===")

    present = subjects_with_all_movies(data_dir) if stage == "full-movies" else None
    if present is not None:
        print(f"  scanned {data_dir}: {len(present)} subjects with all four movie .set files")

    results, total = {}, 0
    for rel in RELEASES:
        if stage == "labels":
            subjects = build_labels(data_dir, rel)
        else:
            subjects = build_full_movies(data_dir, rel, present)
        total += len(subjects)
        out_path = out_dir / f"{rel}_{suffix}_regen.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(subjects) + "\n")
        if len(subjects) != expected[rel]:
            print(f"  WARNING {rel}: expected {expected[rel]} subjects, got {len(subjects)}")
        results[rel] = subjects
    print(f"  wrote 8 files to {out_dir} (total {total} subjects)")

    if verify:
        print(f"  verification against on-disk originals in {data_dir}:")
        outcomes = [compare(rel, results[rel], data_dir / f"{rel}_{suffix}.txt")
                    for rel in RELEASES]
        if any(o is False for o in outcomes):
            return False
        if all(o is None for o in outcomes):
            print("  no originals present, nothing verified")
            return None
        return True
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["labels", "full-movies", "both"], default="both")
    ap.add_argument("--data-dir", default=str(PROJECT_ROOT / "data"),
                    help="BIDS and phenotype directory (default: ./data)")
    ap.add_argument("--out-dir", default=None,
                    help="where to write the _regen files (default: the data dir)")
    ap.add_argument("--verify", action="store_true",
                    help="diff each rebuilt list against its on-disk original")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else data_dir
    if not data_dir.is_dir():
        sys.exit(f"Data directory not found: {data_dir}")

    stages = ["labels", "full-movies"] if args.stage == "both" else [args.stage]
    outcomes = []
    for stage in stages:
        outcomes.append(run_stage(stage, data_dir, out_dir, args.verify))
        print()

    if args.verify:
        if any(o is False for o in outcomes):
            print("RESULT: MISMATCH. Originals left untouched.")
            sys.exit(1)
        print("RESULT: MATCH" if any(o for o in outcomes) else "RESULT: nothing verified")


if __name__ == "__main__":
    main()
