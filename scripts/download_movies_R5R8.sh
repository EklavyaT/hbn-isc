#!/usr/bin/env bash
# Parallel movie EEG download for HBN releases R5-R8.
# 4 releases x 4 parallel xargs each = 16 concurrent S3 cp streams.
# Per-subject only re-downloads if movie .set files are missing.

set -u

cd "$(dirname "$0")/.." || exit 1

mkdir -p logs

download_one_subject() {
    local subject=$1
    local release=$2
    local sub_dir="data/${subject}/eeg"
    mkdir -p "$sub_dir"
    # Skip if any movie .set file already exists for this subject
    if ls "$sub_dir"/*task-DespicableMe*.set "$sub_dir"/*task-DiaryOfAWimpyKid*.set \
          "$sub_dir"/*task-FunwithFractals*.set "$sub_dir"/*task-ThePresent*.set 2>/dev/null | grep -q .; then
        return 0
    fi
    aws s3 cp "s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_${release}/${subject}/eeg/" \
        "$sub_dir/" \
        --recursive --no-sign-request \
        --exclude '*' \
        --include '*DespicableMe*' \
        --include '*DiaryOfAWimpyKid*' \
        --include '*FunwithFractals*' \
        --include '*ThePresent*' \
        --include '*coordsystem*' \
        --quiet
}
export -f download_one_subject

for R in R5 R6 R7 R8; do
    (
        echo "[$(date '+%H:%M:%S')] Starting $R download (n subjects = $(wc -l < data/${R}_subjects_with_labels.txt))"
        cat "data/${R}_subjects_with_labels.txt" | \
            xargs -P 4 -I {} bash -c 'download_one_subject "$@"' _ {} "$R"
        echo "[$(date '+%H:%M:%S')] Done $R"
    ) > "logs/download_${R}.log" 2>&1 &
done

wait
echo "[$(date '+%H:%M:%S')] All downloads complete."
du -sh data/
