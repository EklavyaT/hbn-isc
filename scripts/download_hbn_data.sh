#!/usr/bin/env bash
# HBN EEG downloader for this analysis.
#
# Fetches BIDS EEG recordings from the public FCP-INDI S3 bucket, plus the
# per-release participants.tsv phenotype files that the master table builder
# needs. Nothing here is machine specific: all paths are relative to the repo
# root, and the bucket is read with --no-sign-request (no credentials needed).
#
# Running with no arguments reproduces the original behavior of this script
# exactly: the four movie tasks for releases R5 to R8.
#
# Usage:
#   bash scripts/download_hbn_data.sh
#       Movie tasks, releases R5 to R8. The historical default.
#
#   bash scripts/download_hbn_data.sh --tasks resting --releases R1 \
#        --subject-list data/all_R1_to_process.txt
#       RestingState for the R1 resting cohort. Needed for Figure 2 and the
#       resting row of Table 2.
#
#   bash scripts/download_hbn_data.sh --tasks all \
#        --releases "R1 R2 R3 R4 R5 R6 R7 R8"
#       Everything the full pipeline needs.
#
# Options:
#   --tasks         movies | resting | all           (default: movies)
#   --releases      space separated release tags     (default: "R5 R6 R7 R8")
#   --subject-list  file with one subject ID per line, applied to every release
#                   in this run. Default per release: data/{REL}_subjects_with_labels.txt
#   --parallel      concurrent downloads per release (default: 4)
#
# Task labels are the BIDS task entities as they appear on S3, verified against
# the bucket: task-DespicableMe, task-DiaryOfAWimpyKid, task-FunwithFractals,
# task-ThePresent, task-RestingState. Each .set is self contained; the R1
# recordings carry no companion .fdt file.

set -u

cd "$(dirname "$0")/.." || exit 1
mkdir -p logs data

TASKS="movies"
# All eight releases are needed for the 1535 subject preprocessing set and the
# 1143 subject analytic cohort. The subject lists for every release ship in
# data/, so this default works from a clean clone.
RELEASES="R1 R2 R3 R4 R5 R6 R7 R8"
SUBJECT_LIST=""
PARALLEL=4

while [ $# -gt 0 ]; do
    case "$1" in
        --tasks)        TASKS="$2"; shift 2 ;;
        --releases)     RELEASES="$2"; shift 2 ;;
        --subject-list) SUBJECT_LIST="$2"; shift 2 ;;
        --parallel)     PARALLEL="$2"; shift 2 ;;
        -h|--help)      sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

MOVIE_TASKS="DespicableMe DiaryOfAWimpyKid FunwithFractals ThePresent"
RESTING_TASKS="RestingState"

case "$TASKS" in
    movies)  TASK_LIST="$MOVIE_TASKS" ;;
    resting) TASK_LIST="$RESTING_TASKS" ;;
    all)     TASK_LIST="$MOVIE_TASKS $RESTING_TASKS" ;;
    *) echo "Unknown --tasks value: $TASKS (expected movies, resting, or all)" >&2; exit 2 ;;
esac

export TASK_LIST

echo "tasks    : $TASK_LIST"
echo "releases : $RELEASES"
echo "parallel : $PARALLEL per release"

download_one_subject() {
    local subject=$1
    local release=$2
    local sub_dir="data/${subject}/eeg"
    mkdir -p "$sub_dir"

    # Skip this subject if a .set already exists for every requested task.
    local have_all=1
    for task in $TASK_LIST; do
        if ! ls "$sub_dir"/*task-${task}*.set >/dev/null 2>&1; then
            have_all=0
            break
        fi
    done
    if [ "$have_all" -eq 1 ]; then
        return 0
    fi

    local include_args=()
    for task in $TASK_LIST; do
        include_args+=(--include "*task-${task}*")
    done

    aws s3 cp "s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_${release}/${subject}/eeg/" \
        "$sub_dir/" \
        --recursive --no-sign-request \
        --exclude '*' \
        "${include_args[@]}" \
        --include '*coordsystem*' \
        --quiet
}
export -f download_one_subject

for R in $RELEASES; do
    # The phenotype file is small and is required by scripts/build_master_table.py
    # and scripts/build_r1_subject_list.py. Fetch it once per release.
    if [ ! -f "data/${R}_participants.tsv" ]; then
        aws s3 cp "s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_${R}/participants.tsv" \
            "data/${R}_participants.tsv" --no-sign-request --quiet \
            && echo "fetched data/${R}_participants.tsv"
    fi
done

for R in $RELEASES; do
    if [ -n "$SUBJECT_LIST" ]; then
        LIST="$SUBJECT_LIST"
    else
        LIST="data/${R}_subjects_with_labels.txt"
    fi
    if [ ! -f "$LIST" ]; then
        echo "[$R] SKIP: subject list not found at $LIST" >&2
        continue
    fi
    (
        echo "[$(date '+%H:%M:%S')] Starting $R download (n subjects = $(wc -l < "$LIST"))"
        xargs -P "$PARALLEL" -I {} bash -c 'download_one_subject "$@"' _ {} "$R" < "$LIST"
        echo "[$(date '+%H:%M:%S')] Done $R"
    ) > "logs/download_${R}.log" 2>&1 &
done

wait
echo "[$(date '+%H:%M:%S')] All downloads complete."
du -sh data/
