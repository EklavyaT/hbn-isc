# Repository audit

Audit of the analysis codebase for the HBN EEG ISC manuscript. The working
project was never modified except by adding new builder scripts; every file here
is a copy, and no original script or data file was edited, moved, or deleted.

**Status: every processing stage has a script, and the code chain is unbroken
from raw HBN recordings through to every manuscript table and figure.** All six
previously missing intermediate builders were reconstructed and each reproduces
its original file exactly. The script ambiguity is resolved, the hardcoded paths
are fixed, and the resting-state producer has been staged, closing the last code
gap. What remains unscripted is data acquisition, detailed under "Remaining
non-blocking items".

Verification is of two kinds and they are labeled throughout: the six builders
were verified by **execution** against ground-truth files, while the
resting-state branch was verified by **dependency tracing** because its
preprocessed inputs were not on disk at audit time.

## Verification standard

Each builder regenerates its target to a `_regen` filename and diffs against the
original. Originals were treated as ground truth and never overwritten. A match
means identical subject sets, identical column sets and order, and numeric
agreement to within 1e-9 (observed agreement was 1e-15 or better, consistent with
a float round-trip through CSV).

## Builder reconstruction results

| Builder | Target file | Rows x cols | Result |
|---|---|---|---|
| `build_master_table.py` | `R12345678_master.csv` | 1535 x 35 | **MATCH**, all 35 columns identical |
| `build_isc_cohort.py` | `R12345678_isc_cohort.csv` | 1143 x 35 | **MATCH**, all 35 columns identical |
| `build_band_subsample.py` | `R12345678_band_subsample.csv` | 400 x 4 | **MATCH**, 400/400 subjects, full frame identical |
| `build_sex_analysis_df.py` | `R12345678_sex_analysis_df.csv` | 4568 x 16 | **MATCH**, all 16 columns identical |
| `build_sex_effect_per_channel.py` | `R12345678_sex_effect_per_channel.csv` | 516 x 10 | **MATCH**, max abs diff 1.8e-15 |
| `build_sex_effect_per_channel.py` | `R12345678_sex_effect_theta_per_channel.csv` | 516 x 10 | **MATCH**, max abs diff 4.4e-16 |

Six of six reproduce exactly.

### Subsample seed-42 verification (the item with a real chance of failing)

**Verified.** The manuscript describes the balanced 400 subject subsample as
sampled within sex with seed 42. That description is correct and reproducible.

The exact construction is per-sex `pandas.DataFrame.sample(n=200,
random_state=42)` drawn from the ISC cohort, then sorted by participant_id. A
systematic search over 28 seeding variants was run before settling on this:
three variants reproduce the file exactly (per-sex `pandas.sample`, in either sex
order, and legacy `np.random.seed(42)` reseeded per sex, which is the same
underlying generator). The next best alternative recovered only 309 of 400
subjects, so the match is unambiguous rather than coincidental.

Two details are load bearing and are documented in the builder:

1. The pool must be the cohort file in its **native row order**. Sorting the pool
   before sampling changes which subjects are drawn (that variant recovers only
   168 of 400).
2. The seed is applied **per sex**, not once for the whole draw. Seeding once and
   drawing both sexes from the same generator recovers 309 of 400.

Anyone reproducing this must therefore preserve cohort row order, which the
pipeline does by default.

### Notable finding during reconstruction

The ISC cohort criterion is not simply "all four movies passed QC". Filtering on
`n_movies_pass == 4` plus the QC thresholds and complete CBCL yields **1145**
subjects, two more than the stored cohort of 1143. Two subjects
(`sub-NDARCU736GZ1`, `sub-NDARPE752VYE`) had movies that failed to process while
every movie that did process passed QC, so they carry `n_movies_pass == 4` but
were excluded. Requiring `n_movies_failed == 0` reproduces the cohort exactly.
This is now explicit in `build_isc_cohort.py` rather than implicit in a
precomputed flag column.

## Movie subject list reconstruction

The eight `{REL}_subjects_full_movies.txt` files define the 1535 subject set
submitted to preprocessing. They were reconstructed and verified by **execution**
against the on-disk originals, matching on both count and membership for every
release.

**The rule is file presence, not QC and not the phenotype flags.** A subject
enters the list when all four movie `.set` files are present on disk after
download, intersected with that release's `participants.tsv` membership. Two
candidate rules were tested and rejected first:

| Candidate | Result |
|---|---|
| `participants.tsv` movie availability flags, with or without CBCL | **Rejected.** 1544 or 1516 against 1535; membership wrong for R3 to R8. The flags do not define the cohort. |
| `R12345678_master.csv` membership | Matches all eight exactly, but **circular**: the master is built from preprocessing results that require these lists as input, so it cannot serve as a builder source on a clean clone. |
| On-disk presence of all four movie `.set` files | **Accepted.** All eight exact, by count and membership, totalling 1535. |

A second list family was also reconstructed: `{REL}_subjects_with_labels.txt`,
which drives the download, is exactly the participants with complete CBCL
bifactor scores (1867 across eight releases, all eight exact).

`scripts/build_movie_subject_lists.py` regenerates both families, writes to
`_regen` filenames so it can never overwrite the shipped copies, and reads no
pipeline output. Per-release verification results:

| Stage | Counts | Verified |
|---|---|---|
| `labels` | 132, 147, 180, 318, 323, 131, 379, 257 (1867) | 8 of 8 exact |
| `full-movies` | 120, 120, 157, 293, 282, 102, 247, 214 (1535) | 8 of 8 exact |

One caveat, documented in the builder and the README rather than glossed: the
`full-movies` stage enumerates what is on disk, so it reproduces 1535 only when
the download covered the shipped `with_labels` sets. A broader download could
enumerate more subjects. The shipped lists are authoritative.

## Resting-state acquisition closure

The resting branch is now acquirable and runnable from a clean clone. Three
additions, all in the staged repository only:

| Addition | Purpose | Verification |
|---|---|---|
| `download_hbn_data.sh` | Fetches any release and task group, including `task-RestingState`, plus `participants.tsv`. Defaults reproduce the old movie-only behavior exactly. | Filter logic confirmed by `aws s3 cp --dryrun` against the live bucket: the resting filter selects `{sid}_task-RestingState_eeg.set`, the exact file `run_v3_batch_R1.py` opens. |
| `build_r1_subject_list.py` | Regenerates `data/all_R1_to_process.txt` from the R1 phenotype file. | Selects R1 participants with complete CBCL scores: exactly 132, identical to both `R1_results.csv` (the batch's own log, 0 failed, 0 skipped) and the 132 in `R1_resting_state_isc.csv`. Zero extras, zero missing. |
| `data/all_R1_to_process.txt` | The 132 subject IDs, shipped so the branch runs without rebuilding. Subject IDs only, no phenotype data. | Byte-identical to the builder's output. |

The BIDS naming was confirmed against the live bucket rather than assumed: the
task label is `RestingState` and the recording is
`{subject}_task-RestingState_eeg.set`, with no companion `.fdt`.

`participants.tsv` is deliberately **not** redistributed in this repository. It
carries participant-level age and CBCL psychopathology scores, so the downloader
fetches it instead.

An 11 point trace confirms the branch connects end to end: downloader task label
and destination directory, batch subject list and raw path, batch output
directory and suffix against script 14's glob, and script 14's output against
both consumers. All 11 links match exactly.

## Resting-state producer reclassification

`run_v3_batch_R1.py` was previously classified EXCLUDE_SUPERSEDED and is now
**INCLUDE**. It is superseded for the movie pipeline, which uses the
concatenated-ICA path, but it is the **sole producer** of the RestingState
preprocessed files that `14_isc_resting_state.py` consumes, and therefore of the
baseline behind Figure 2 and the resting row of Table 2. It is self-contained,
with its own inline `process_subject`, and contains no hardcoded absolute paths,
so it was staged unmodified.

`preprocess_one_subject_v3.py` was also moved to **INCLUDE**, but on weaker
grounds and with a correction to the original premise for staging it: **it is not
a producer of the cohort RestingState files.** It is hardcoded to a single
subject (`sub_NDARAC904DMU`) and writes to `outputs/` root rather than
`preprocessed_R1/`, so it is a single-subject validation prototype, not a batch
stage. `run_v3_batch_R1.py` does not import it and reimplements the steps inline.
It is included for provenance only and the README lists it as standalone, outside
the run order. Excluding it would not break any chain.

Neither script required a path fix. Unlike scripts 16, 16b, and 18, both use only
`PROJECT_ROOT`-relative paths.

The two pipelines use deliberately different output conventions, which is not a
defect: movies write `preprocessed_movies_{REL}/*_preproc_v3concat_raw.fif`,
resting writes `preprocessed_R1/*_RestingState_preproc_v3_raw.fif`.

## Script 16 versus 16b resolution

**Canonical: `16b_isc_samesex_broadband_gram.py`.** Script 16 is superseded and
is not included in this repository.

Both scripts implement Gate 1 Test B (broadband same-sex size-matched ISC) and
write to identical output paths, so only one could have produced the stored
results. Three independent lines of evidence identify 16b:

1. `outputs/R12345678_testB_draw_sd.csv` records `B_draws=50` for all four
   movies. Script 16b sets `B_DRAWS = 50`; script 16 sets `B_DRAWS = 20`, with an
   inline comment saying it was "reduced from 50" because the memmap approach was
   I/O bound. The published methods state 50 draws.
2. The mean per-male draw count in the stored output is 24.71, which matches the
   expectation for 50 draws (50 x 378/765 = 24.7). Twenty draws would give 9.9.
3. Script 16b (modified Jun 24 00:44) predates the stored outputs (Jun 24 07:51);
   script 16 was last modified Jun 23 21:13, before 16b existed.

Running the downstream `samesex_testB_stats.py` on the stored files reproduces
all four published Gate 1 Test B values exactly: whole-head grand mean Hedges g
0.873 pooled to 1.140 same-sex size-matched, and frontocentral cluster g 1.376 to
1.751. The Test B result is therefore confirmed to come from the 50-draw Gram
implementation.

**Minor manuscript discrepancy worth checking.** The published text describes the
male template as subsampled to "female n=377". The stored per-movie record shows
`K_matched = 378` for Despicable Me, Diary of a Wimpy Kid, and Fun with Fractals,
and 377 only for The Present, which loses one subject to the script 12
minimum-duration patch. The text should either say 378, or say "377 to 378
depending on movie". This does not affect any reported statistic.

## Path fixes

Applied to the staged copies only; the originals in the working project are
unchanged and still contain their hardcoded paths.

| Script | Was | Now |
|---|---|---|
| `18_gate2_refit.py` | `/Volumes/PortableSSD/Projects/HBN-BrainAI/outputs/gate2_cache` | `HBN_GATE2_CACHE` env var, default `./outputs/gate2_cache` |
| `16b_isc_samesex_broadband_gram.py` | `/Volumes/PortableSSD/Projects/HBN-BrainAI` | `HBN_SCRATCH_ROOT` env var, default `./scratch` |

Two follow-on fixes were required and made:

- Script 16b's `ssd_available()` probed `SSD_ROOT / "outputs"`, which was
  meaningful only for the external SSD layout. It now creates and checks the
  scratch root itself, so the transient-unmount retry logic degrades cleanly to a
  no-op on a fixed local disk.
- Script 18 used `os.environ` after the edit but did not import `os`. This passes
  `py_compile` and would have failed only at runtime with a `NameError`. The
  import was added. Both scripts were re-checked for import and syntax validity.

No hardcoded absolute paths remain anywhere in the staged repository.

## Remaining non-blocking items

These do not prevent reproduction of any manuscript table or figure.

**R1. Figure 1 has no generating script.** `figures/paper/` contains Figures 2
through 5 and S1 through S5. Figure 1 is a conceptual schematic produced outside
the codebase. The README states this. Add the source file if one exists.

**R2. Resolved. All subject lists now ship and are regenerable.** See "Movie
subject list reconstruction" below. The downloader covers all eight releases for
both task groups by default, and every list a pipeline stage reads is present in
`data/`.

**R3. `decord` is imported but was not installed** in the audited environment.
`extract_clip_features.py` imports it for video frame sampling. Either that path
was run in a different environment or never exercised as written. Left unpinned
in `requirements.txt` with a note. Affects the Figure 5B encoding model only.

**R4. Directory preconditions.** Scripts assume `data/` and `outputs/` exist
beside `scripts/`. Nothing creates them, so a first run fails with a bare
`FileNotFoundError`. A `mkdir -p` at each entry point would fix this.

**R5. One REVIEW file remains unstaged and undecided.**
`01_build_master_table.py` (R1-only master, now functionally superseded by
`build_master_table.py`). It does not block reproduction.

**R7. The resting-state branch was verified by tracing, not execution.** The
preprocessed RestingState FIFs were absent from local disk at audit time and the
external volume was unmounted, so `14_isc_resting_state.py` could not be run.
What was verified: `run_v3_batch_R1.py` writes
`outputs/preprocessed_R1/{sid}_RestingState_preproc_v3_raw.fif`, and script 14
globs that identical directory and suffix, so the chain connects. Separately, all
five resting values reported in the paper were confirmed against the existing
`R1_resting_state_isc.csv` (n=132, mean +0.0013, SD 0.0101, t 1.51, p 0.133,
per-channel range -0.0113 to +0.0059). An end-to-end execution of the resting
branch on restored raw data would close this to the same standard as the six
builders.

**R6. The builders write to `_regen` filenames by design**, so a clean-room run
produces `R12345678_master_regen.csv` rather than `R12345678_master.csv`. This
was deliberate, to guarantee no existing file could be overwritten during
verification. A first-time user must rename, or a small flag should be added to
write the canonical name when no original is present.

## Final classification

| Category | Count |
|---|---|
| INCLUDE (staged) | 42 |
| REVIEW (undecided, unstaged) | 1 |
| EXCLUDE_SUPERSEDED | 16 |
| EXCLUDE_SCRATCH | 3 |
| **Total code files in source project** | **59** |

The source project grew from 54 to 59 code files: the five new builder scripts
were added there as well, since `build_sex_effect_per_channel.py` produces two of
the six target files.

Changes in this revision: `run_v3_batch_R1.py` and `preprocess_one_subject_v3.py`
moved from EXCLUDE_SUPERSEDED to INCLUDE (see "Resting-state producer
reclassification"), and `trf_batch.py` moved from REVIEW to EXCLUDE_SUPERSEDED
after the encoding provenance check confirmed every reported TRF value comes from
the banded run. `01_build_master_table.py` remains the sole REVIEW file: it is
functionally superseded by `build_master_table.py`, but it is the R1-era original
and you may want it kept for provenance, so that call is left to you.

## Artifact to producer mapping, resting-state branch

| Artifact | Producing chain |
|---|---|
| Figure 2, ISC overview | `run_v3_batch_R1.py` then `14_isc_resting_state.py` for the resting baseline; `11_compute_isc_R12345678.py` plus `12_patch_ThePresent_isc.py` for the movie ISC; rendered by `figures/figure2.py` |
| Table 2 resting row | same resting chain, consumed at `build_tables.py` in `table2_isc_magnitudes` |
| `R1_resting_state_isc.csv` | `14_isc_resting_state.py`, reading `outputs/preprocessed_R1/*_RestingState_preproc_v3_raw.fif` from `run_v3_batch_R1.py` |

The full mapping for all other tables and figures is in `README.md`.

## Verdict

**Yes for the code and the inputs, with one honest note on how the final stages
were verified.** From a clean clone a user can fetch every input with
`download_hbn_data.sh` (all eight releases, movie and resting tasks, plus the
per-release `participants.tsv`) and run through to every manuscript table and
figure. No pipeline stage reads a file that is neither downloadable from the
public bucket nor produced by a staged script nor shipped in `data/`.

What was verified by execution: the six intermediate data builders, each
reproducing its original file exactly; the balanced subsample's seed-42
construction; the R1 resting cohort list against two independent ground truths;
and all sixteen movie subject lists, matching on both count and membership across
eight releases. What was verified by dependency tracing rather than execution:
the two preprocessing branches themselves, since rerunning them means days of
compute, and in the resting case the preprocessed files were not on disk. The
download filters were additionally checked with a live `aws s3 cp --dryrun`
against the bucket. Published values were confirmed against existing outputs
where a stage could not be rerun.

Also settled: the 16 versus 16b ambiguity, with three independent lines of
evidence; the encoding numbers, confirmed to come from the banded run; no
hardcoded absolute paths; and no participant-level phenotype data redistributed
here, only subject IDs.

The remaining items are not code. A human must choose and add a LICENSE file, and
install and pin `decord` if the Figure 5B encoding analysis is to be run. Two
smaller items carry over: Figure 1 is an external schematic with no source file,
and the manuscript should be corrected to say the Test B male template was size
matched to 378 for three movies and 377 for The Present, not 377 throughout.
