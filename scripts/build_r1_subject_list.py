"""Build the R1 subject list that the resting-state preprocessing iterates.

Reconstruction of data/all_R1_to_process.txt, which scripts/run_v3_batch_R1.py
reads but which no script previously generated. Verified to reproduce the exact
132 subject set that the published resting-state analysis used.

Selection rule: R1 participants with complete CBCL bifactor scores, that is,
p_factor, attention, internalizing, and externalizing all non-null. This matches
the description in run_v3_batch_R1.py ("the full R1 cohort, 132 subjects with
CBCL labels") and yields exactly 132 of the 136 R1 participants, with no extras
and none missing.

Source: data/R1_participants.tsv, the phenotype file shipped with the R1 BIDS
release. scripts/download_hbn_data.sh fetches it. The rule deliberately does NOT
derive the list from any pipeline output (for example outputs/R1_results.csv or
outputs/R1_resting_state_isc.csv), because those are produced by the very
pipeline this list feeds, which would be circular on a clean clone.

Note that the participants file marks RestingState as available for all 136 R1
subjects, so filtering on recording availability is redundant here and is not
applied. If a future release has missing RestingState entries, add that
condition.

Usage:
    python scripts/build_r1_subject_list.py
    python scripts/build_r1_subject_list.py --verify   # diff against an existing list
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARTICIPANTS = PROJECT_ROOT / "data" / "R1_participants.tsv"
OUT = PROJECT_ROOT / "data" / "all_R1_to_process.txt"

CBCL = ["p_factor", "attention", "internalizing", "externalizing"]
EXPECTED_N = 132


def build():
    if not PARTICIPANTS.exists():
        sys.exit(
            f"Participants file not found: {PARTICIPANTS}\n"
            "Fetch it with:\n"
            "  bash scripts/download_hbn_data.sh --tasks resting --releases R1"
        )
    p = pd.read_csv(PARTICIPANTS, sep="\t")
    missing = [c for c in CBCL if c not in p.columns]
    if missing:
        sys.exit(f"Participants file is missing expected columns: {missing}")
    keep = p[p[CBCL].notna().all(axis=1)]
    return sorted(keep["participant_id"].astype(str).tolist())


def verify(subjects):
    """Compare against an existing list if one is present."""
    if not OUT.exists():
        print("  no existing list to compare against, skipping verification")
        return None
    existing = [s.strip() for s in OUT.read_text().splitlines() if s.strip()]
    same = set(existing) == set(subjects)
    print(f"  existing n={len(existing)}  rebuilt n={len(subjects)}")
    print(f"  overlap: {len(set(existing) & set(subjects))}/{len(existing)}")
    print(f"  identical subject set: {same}")
    if not same:
        only_new = sorted(set(subjects) - set(existing))
        only_old = sorted(set(existing) - set(subjects))
        for s in only_new[:5]:
            print(f"    only in rebuilt : {s}")
        for s in only_old[:5]:
            print(f"    only in existing: {s}")
    return same


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="diff the rebuilt list against an existing one")
    args = ap.parse_args()

    subjects = build()
    print(f"selected {len(subjects)} R1 subjects with complete CBCL scores")
    if len(subjects) != EXPECTED_N:
        print(f"WARNING: expected {EXPECTED_N} subjects, got {len(subjects)}. "
              "The published resting-state analysis used 132. Investigate before "
              "proceeding rather than assuming the difference is benign.")

    if args.verify:
        print("\nverification against the existing list:")
        ok = verify(subjects)
        if ok is False:
            print("\nRESULT: MISMATCH. Existing list left untouched.")
            sys.exit(1)
        print("\nRESULT: MATCH" if ok else "\nRESULT: no baseline to compare")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(subjects) + "\n")
    print(f"wrote {OUT} ({len(subjects)} subjects)")


if __name__ == "__main__":
    main()
