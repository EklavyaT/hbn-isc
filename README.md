# Sex differences in pediatric EEG inter-subject correlation (HBN-EEG)

Analysis code for an IEEE manuscript on sex differences in inter-subject
correlation (ISC) of pediatric EEG recorded during naturalistic movie watching,
using the public Healthy Brain Network (HBN) EEG dataset. The reported effect is
higher ISC in males than females, localized to a frontocentral cluster (channels
E30, E36, E37 on the GSN-HydroCel-129 montage), strongest in the delta band (1 to
4 Hz) and then theta (4 to 8 Hz), and null in alpha and beta. The primary cohort
is 1143 subjects across HBN releases R1 to R8, with a balanced 400 subject
subsample (200 male, 200 female) used for the frequency-band analyses. This
repository contains the preprocessing pipeline, the ISC computation, the
frequency-band decomposition, two validity gates (same-sex templates and ocular
artifact), the statistics, and the figure and table generation. It contains no
data.

The pipeline reproduces every manuscript table and figure from the public HBN
data. The six intermediate builder scripts were each verified to regenerate their
original file exactly, and the resting-state branch used by Figure 2 and Table 2
has its producer included. What is not fully scripted is data acquisition: the
download helper covers only part of what the pipeline needs, as described under
"Data" and "Known gaps". `REPO_AUDIT.md` holds the full audit and verification
results.

## Data

The HBN-EEG data is public and is **not** included here. Download it separately
from the FCP-INDI S3 bucket:

```
s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_{RELEASE}/{SUBJECT}/eeg/
```

where `{RELEASE}` is `R1` through `R8`. **All eight releases are required for the
full n=1143 cohort.** The four movie-watching tasks used are `DespicableMe`,
`DiaryOfAWimpyKid`, `FunwithFractals`, and `ThePresent`.

`scripts/download_hbn_data.sh` fetches everything the pipeline needs from S3:
any release, either task group, plus the per-release `participants.tsv` phenotype
files that `build_master_table.py` requires. It needs no credentials.

```
bash scripts/download_hbn_data.sh                                    # movie tasks, all 8 releases
bash scripts/download_hbn_data.sh --tasks resting --releases R1 \
     --subject-list data/all_R1_to_process.txt                       # resting-state branch
```

The default now covers releases R1 to R8 for the movie tasks, which is what the
1535 subject preprocessing set requires. Both commands together fetch every input
the pipeline needs.

The task labels were verified against the bucket: `task-DespicableMe`,
`task-DiaryOfAWimpyKid`, `task-FunwithFractals`, `task-ThePresent`, and
`task-RestingState`. Each `.set` is self contained, with no companion `.fdt`.
The original `download_movies_R5R8.sh` is retained unchanged for reference; the
new helper reproduces its behavior exactly when run with no arguments.

**Subject lists.** All subject lists ship with this repository (subject IDs
only, no phenotype data), and all are regenerable:

| List | Count | Consumed by | Rebuilt by |
|---|---|---|---|
| `{REL}_subjects_with_labels.txt` | 1867 across 8 releases | the downloader | `build_movie_subject_lists.py --stage labels` |
| `{REL}_subjects_full_movies.txt` | 1535 across 8 releases | `run_movies_v3concat_batch.py` | `build_movie_subject_lists.py --stage full-movies` |
| `all_R1_to_process.txt` | 132 | `run_v3_batch_R1.py` | `build_r1_subject_list.py` |

The `with_labels` lists select participants with complete CBCL bifactor scores.
The `full_movies` lists select subjects that have all four movie `.set` files
present after download, which is a file-presence rule rather than a QC rule: it
defines the 1535 subject set submitted to preprocessing, which the QC filter in
`build_isc_cohort.py` later narrows to the 1143 subject analytic cohort.

`participants.tsv` is fetched by the helper into `data/{RELEASE}_participants.tsv`
and is **not** redistributed here, since it carries participant-level age and CBCL
psychopathology scores.

Movie stimulus files (the `.mp4` videos) are needed only for the encoding model in
Figure 5B. They are distributed by the Child Mind Institute separately from the
EEG data and are not on the S3 bucket above.

## Environment

Dependencies and pinned versions are in `requirements.txt`. There is no
`environment.yml` in this repository.

```
conda create -n hbn_isc python=3.11.15
conda activate hbn_isc
pip install -r requirements.txt
```

Key pinned versions, which are those the manuscript analyses were run under:

| Package | Version |
|---|---|
| Python | 3.11.15 |
| MNE | 1.12.0 |
| PyPREP | 0.6.0 |
| MNE-ICALabel | 0.8.1 |
| NumPy | 2.4.4 |
| pandas | 3.0.2 |
| SciPy | 1.17.1 |
| scikit-learn | 1.8.0 |
| statsmodels | 0.14.6 |
| matplotlib | 3.10.8 |

MNE and MNE-ICALabel versions matter: ICA component labels can differ across
minor versions, which changes which components are excluded during preprocessing
and therefore changes downstream results.

`decord` is imported by `scripts/extract_clip_features.py` for video frame
sampling, but it was **not present in the audited environment**, so its version is
unverified and it is left unpinned in `requirements.txt`. Install it separately if
you intend to run the encoding analysis. It affects Figure 5B only.

## System requirements

- Roughly 1 TB free disk for the raw downloads plus preprocessed FIF files across
  R1 to R8.
- 16 GB RAM minimum. The ISC scripts use a multi-pass streaming design
  specifically to avoid holding the full (subjects, channels, timepoints) tensor
  in memory.
- Preprocessing all four movies for the full cohort is the dominant cost and runs
  on the order of days on a single machine. The batch runner is idempotent and
  skips subjects whose outputs already exist, so runs can be interrupted and
  resumed.
- A GPU is optional and benefits only the Figure 5B encoding model.
- `data/` and `outputs/` must exist beside `scripts/`. Nothing in the code creates
  them, and a first run without them fails with a bare `FileNotFoundError`.

Two scratch locations are configurable by environment variable:

| Variable | Used by | Default |
|---|---|---|
| `HBN_GATE2_CACHE` | `18_gate2_refit.py` | `./outputs/gate2_cache` |
| `HBN_SCRATCH_ROOT` | `16b_isc_samesex_broadband_gram.py` | `./scratch` |

Both grow large. Point them at a disk with room.

## The `_regen` filename convention (read before running)

The six `build_*` scripts write to filenames ending in `_regen`, for example
`outputs/R12345678_master_regen.csv`. This was deliberate: it guarantees a
verification run can never overwrite an existing ground-truth file.

**Every downstream script hardcodes the name without `_regen`.** The chain
therefore does not connect on a clean run until you rename. After running the
builders, perform exactly these six renames:

```
mv outputs/R12345678_master_regen.csv                     outputs/R12345678_master.csv
mv outputs/R12345678_isc_cohort_regen.csv                 outputs/R12345678_isc_cohort.csv
mv data/R12345678_band_subsample_regen.csv                data/R12345678_band_subsample.csv
mv outputs/R12345678_sex_analysis_df_regen.csv            outputs/R12345678_sex_analysis_df.csv
mv outputs/R12345678_sex_effect_per_channel_regen.csv     outputs/R12345678_sex_effect_per_channel.csv
mv outputs/R12345678_sex_effect_theta_per_channel_regen.csv outputs/R12345678_sex_effect_theta_per_channel.csv
```

Note that `build_band_subsample.py` writes into `data/`, the other five into
`outputs/`. Rename each builder's output immediately after running it, because
later builders read the canonical names of earlier ones: `build_isc_cohort.py`
reads `R12345678_master.csv`, and `build_sex_analysis_df.py` reads
`R12345678_isc_cohort.csv`. Pointing downstream scripts at the `_regen` files
instead would require editing the path constant at the top of each consumer, so
renaming is the simpler route.

The `--verify` flag on each builder diffs the regenerated file against an existing
original when one is present, and is a silent no-op when none exists. On a clean
machine it is harmless to pass.

## Run order

Derived from the input and output filenames each script actually declares. Every
script in the repository appears below or is listed under "Standalone and library
files".

### 1. Acquire data

```
bash scripts/download_hbn_data.sh                                    # movies, R1 to R8
bash scripts/download_hbn_data.sh --tasks resting --releases R1 \
     --subject-list data/all_R1_to_process.txt
```

Every subject list these commands need ships with the repository, so this runs
from a clean clone. To rebuild the lists instead of using the shipped copies,
after the download completes:

```
python scripts/build_movie_subject_lists.py --verify   # both movie list stages
python scripts/build_r1_subject_list.py --verify       # resting cohort list
```

`build_movie_subject_lists.py` writes to `_regen` filenames and diffs against the
shipped copies, so it can never overwrite them. Its `full-movies` stage
enumerates the downloaded BIDS tree, so run it after the download, and note that
it reproduces the published 1535 only when the download covered the shipped
`with_labels` sets. The shipped `full_movies` lists remain authoritative for
exact reproduction.

### 2. Preprocess

```
python scripts/build_movie_subject_lists.py --verify        # optional, rebuild lists
python scripts/run_movies_v3concat_batch.py --release R1    # repeat for R1 to R8
python scripts/run_v3_batch_R1.py                          # resting state, R1 only
```

`build_movie_subject_lists.py` is optional because the lists it produces already
ship. Run it only to regenerate them from the downloaded BIDS tree; it must run
after the download and before the batch runners.

Batch runner over one release; calls `preprocess_movies_v3_concat.py`, the
published pipeline (crop to the movie window, 60/120 Hz notch, split into a 1 to
100 Hz ICA copy and a 1 to 20 Hz analysis copy, common average reference, PyPREP
bad-channel detection and interpolation, one ICA fit on the concatenation of all
four movies, ICALabel classification, per-movie component removal, re-reference,
resample to 200 Hz). Writes
`outputs/preprocessed_movies_{REL}/{subject}_{task}_preproc_v3concat_raw.fif` and
the per-release QC file `outputs/{REL}_movies_results.csv`.

`run_v3_batch_R1.py` is the separate resting-state preprocessing batch. It applies
the same v3 steps (notch, dual filter split, average reference, PyPREP, ICA with
ICALabel component removal, resample to 200 Hz) to the R1 `task-RestingState`
recordings, and writes
`outputs/preprocessed_R1/{subject}_RestingState_preproc_v3_raw.fif` plus
`outputs/R1_results.csv`. Note the deliberate two-pipeline convention: the movie
pipeline writes `preprocessed_movies_{REL}/` with a `_preproc_v3concat_raw.fif`
suffix, while the resting pipeline writes `preprocessed_R1/` with a
`_RestingState_preproc_v3_raw.fif` suffix. Script 14 in step 5 reads the latter.

### 3. Cohort tables

```
python scripts/build_master_table.py --verify   # then rename _regen
python scripts/build_isc_cohort.py --verify     # then rename _regen
```

- `build_master_table.py`: aggregates the per-release QC results and joins the
  `participants.tsv` phenotype files. Produces the 1535 subject master table.
- `build_isc_cohort.py`: applies loose QC (std_p50 <= 40, ICALabel confidence
  >= 0.50, brain ICs >= 3), requires all four movies processed without failure and
  passing QC, and requires complete CBCL scores. Produces the 1143 subject cohort.

### 4. Broadband ISC

```
python scripts/11_compute_isc_R12345678.py
python scripts/12_patch_ThePresent_isc.py
```

- `11_compute_isc_R12345678.py`: streaming leave-one-out per-channel Pearson ISC
  across the cohort. Writes `R12345678_isc_per_subject.csv` and
  `R12345678_isc_long.csv`.
- `12_patch_ThePresent_isc.py`: required correction, not optional. Script 11's
  global crop was dragged to 57.8 s by four truncated ThePresent recordings. This
  recomputes ThePresent on recordings of at least 200 s and replaces those rows in
  both files. The published numbers include this patch.

### 5. Balanced subsample and band ISC

```
python scripts/build_band_subsample.py --verify   # then rename _regen
python scripts/13_isc_by_band_full.py
python scripts/17_isc_delta_band.py
python scripts/14_isc_resting_state.py
```

- `build_band_subsample.py`: draws the balanced 400 subject subsample from the
  cohort using per-sex `pandas.sample(n=200, random_state=42)`.
- `13_isc_by_band_full.py`: writes `R12345678_isc_by_band_{theta,alpha,beta}.csv`.
- `17_isc_delta_band.py`: writes `R12345678_isc_by_band_delta.csv`.
- `14_isc_resting_state.py`: leave-one-out ISC on the R1 resting-state recordings
  from step 2, giving the near-zero baseline that Figure 2 and Table 2 compare
  against. Writes `outputs/R1_resting_state_isc.csv`.

### 6. Analysis frames

```
python scripts/build_sex_analysis_df.py --verify         # then rename _regen
python scripts/build_sex_effect_per_channel.py --verify  # then rename both _regen files
```

- `build_sex_analysis_df.py`: joins per-subject broadband ISC to the cohort
  covariates, producing the frame every regression is fit on.
- `build_sex_effect_per_channel.py`: per-movie per-channel male versus female
  contrasts (Student two-sample t, BH-FDR within movie across 129 channels) for
  both broadband and theta. Reads `R12345678_isc_per_subject.csv` and
  `R12345678_isc_by_band_theta.csv`, so it must run after step 5.

### 7. Tables and statistics

Run before the validity gates. `samesex_testB_stats.py` in step 8 consumes
`R12345678_sex_effect_extended_stats.csv`, which `extra_analyses.py` writes here.

```
python scripts/tables/build_tables.py
python scripts/tables/extra_analyses.py
python scripts/tables/table6_delta_band.py
python scripts/tables/strict_qc_shrinkage.py
```

- `build_tables.py`: Tables 1 to 6, written as CSV, LaTeX, and Markdown.
- `extra_analyses.py`: revision analyses A1 to A4, B1 to B5 (including the global
  FDR check), C3 and C4.
- `table6_delta_band.py`: integrates the delta band into Table 6. Reads
  `table6_band_breakdown.csv` from `build_tables.py`, so it must run after it.
- `strict_qc_shrinkage.py`: the strict-QC effect-size shrinkage analysis.

Note that `build_tables.py` reads `R1_resting_state_isc.csv` for Table 2. See
"Known gaps".

### 8. Validity gates

Gate 1, same-sex templates (tests whether the male-dominated pooled template
inflates the male side):

```
python scripts/15_isc_samesex_band.py
python scripts/tables/samesex_testA_stats.py
python scripts/16b_isc_samesex_broadband_gram.py
python scripts/tables/samesex_testB_stats.py
```

Gate 2 Tier 1, ocular artifact, spatial and spectral checks:

```
python scripts/tables/gate2_test1_gradient.py
python scripts/tables/gate2_test2_stats.py
python scripts/tables/gate2_delta_topography.py
```

Gate 2 Tier 2, direct pre versus post ocular-component removal:

```
python scripts/18_gate2_refit.py            # re-fits ICA per subject, caches
python scripts/19_gate2_isc_conditions.py   # ISC under conditions A, B, C, OC
python scripts/tables/gate2_tier2_stats.py
```

`18_gate2_refit.py` also writes `data/gate2_balanced150.csv`, the 150 subject
subset it fits, and `19_gate2_isc_conditions.py` reads the ICA cache that 18
produces, so these three run strictly in order.

### 9. Encoding model (Figure 5B only)

```
python scripts/extract_low_level_features.py
python scripts/extract_clip_features.py
python scripts/trf_banded_batch.py
```

Independent of steps 3 through 8; needs only the preprocessed EEG from step 2 and
the movie stimulus files. `trf_banded_batch.py` writes
`trf_banded_30subj_results.csv`, the banded ridge result that Figure 5B uses.

### 10. Figures

```
python scripts/figures/figure2.py
python scripts/figures/figure3.py
python scripts/figures/figure4_sex_headline.py
python scripts/figures/figure5.py
python scripts/figures/supplementary.py
python scripts/figures/figure_typical_development_sex.py
```

Order-independent among themselves. All import `scripts/figures/style.py` for the
shared palette and rcParams. `figure2.py` additionally needs
`R1_resting_state_isc.csv`; see "Known gaps".

### Standalone and library files

- `scripts/preprocess_movies_v3_concat.py`: the preprocessing implementation,
  normally invoked through `run_movies_v3concat_batch.py` in step 2.
- `scripts/figures/style.py`: shared style module, imported by every figure
  script, never run directly.
- `scripts/build_r1_subject_list.py`: regenerates `data/all_R1_to_process.txt`
  from the R1 phenotype file. Run it in step 1 if you want to rebuild the shipped
  list rather than use it as is.
- `scripts/build_movie_subject_lists.py`: regenerates both movie list families.
  Optional, since the lists ship; listed in step 2 because its `full-movies` stage
  reads the downloaded BIDS tree.
- `scripts/download_movies_R5R8.sh`: the original movie-only downloader, retained
  unchanged for reference. `download_hbn_data.sh` supersedes it and reproduces its
  behavior exactly when run with no arguments.
- `scripts/preprocess_one_subject_v3.py`: the v3 pipeline applied to a single
  hardcoded subject, kept as the prototype that documents and validates the step
  sequence. It is **not** part of the run order: it processes one subject and
  writes to `outputs/` root rather than `preprocessed_R1/`. The batch equivalent
  used by the pipeline is `run_v3_batch_R1.py` in step 2.

## Manuscript artifact to script mapping

| Artifact | Script |
|---|---|
| Table 1, cohort demographics | `tables/build_tables.py` (`table1_demographics`) |
| Table 2, ISC magnitudes | `tables/build_tables.py` (`table2_isc_magnitudes`), with the resting row from `run_v3_batch_R1.py` then `14_isc_resting_state.py` |
| Table 3, age regression | `tables/build_tables.py` (`table3_age_regression`) |
| Table 4, sex regression | `tables/build_tables.py` (`table4_sex_regression`) |
| Table 5, topography | `tables/build_tables.py` (`table5_topography`) |
| Table 6, frequency bands | `tables/build_tables.py` (`table6_band_breakdown`) plus `tables/table6_delta_band.py` for the delta row |
| Figure 1 | No generating script. External conceptual schematic, produced outside this codebase. |
| Figure 2, ISC overview | `figures/figure2.py`, with the resting baseline from `run_v3_batch_R1.py` then `14_isc_resting_state.py` |
| Figure 3, developmental effect | `figures/figure3.py` |
| Figure 4, sex effect headline | `figures/figure4_sex_headline.py` |
| Figure 5, CBCL nulls and encoding bound | `figures/figure5.py`, panel B needs `trf_banded_batch.py` |
| Figures S1 to S5 | `figures/supplementary.py` |
| Typical-development sex topography | `figures/figure_typical_development_sex.py` |
| Broadband ISC values | `11_compute_isc_R12345678.py`, `12_patch_ThePresent_isc.py` |
| Band decomposition | `13_isc_by_band_full.py` (theta, alpha, beta), `17_isc_delta_band.py` (delta) |
| Resting-state baseline | `download_hbn_data.sh --tasks resting` (acquisition), `build_r1_subject_list.py` (cohort list), `run_v3_batch_R1.py` (preprocessing), `14_isc_resting_state.py` (ISC) |
| Data acquisition | `download_hbn_data.sh` (all eight releases, movie and resting tasks, plus `participants.tsv`) |
| Movie subject lists (1535 preprocessing set) | `build_movie_subject_lists.py`, shipped at `data/{REL}_subjects_full_movies.txt` |
| Download driver lists (1867) | `build_movie_subject_lists.py --stage labels`, shipped at `data/{REL}_subjects_with_labels.txt` |
| Validity gate 1, same-sex | `15_isc_samesex_band.py`, `16b_isc_samesex_broadband_gram.py`, `tables/samesex_testA_stats.py`, `tables/samesex_testB_stats.py` |
| Validity gate 2, ocular | `18_gate2_refit.py`, `19_gate2_isc_conditions.py`, `tables/gate2_test1_gradient.py`, `tables/gate2_test2_stats.py`, `tables/gate2_delta_topography.py`, `tables/gate2_tier2_stats.py` |
| FDR corrections | `tables/build_tables.py` (per table), `tables/extra_analyses.py` (`run_b5`, global) |
| Strict-QC analysis | `tables/strict_qc_shrinkage.py` |
| Encoding model | `extract_low_level_features.py`, `extract_clip_features.py`, `trf_banded_batch.py` |

## Known gaps

No input gaps remain. Every subject list ships and is regenerable, and the
downloader covers all eight releases for both the movie and resting tasks.

One nuance worth knowing rather than a gap: the `full_movies` rule enumerates
files present on disk, so it reflects what was downloaded. Downloading a broader
subject set than the shipped `with_labels` lists could enumerate additional
subjects and change the cohort. The shipped `full_movies` lists are therefore
authoritative, and the builder writes to `_regen` names so it cannot overwrite
them.

An earlier derivation attempt from the `participants.tsv` availability flags is
worth recording as a negative result: it reproduces R1 and R2 but diverges for R3
through R8 (1516 against 1535, with R4 matching on count while differing in
membership). Those flags do not define the cohort; on-disk file presence does.

**Figure 1 has no generating script.** It is an external schematic.

**`decord` is unverified**, as described under "Environment". Affects Figure 5B.

## Reproducibility notes

- All six `build_*` scripts were verified to regenerate their target file exactly
  against the originals from the published analysis: identical subject sets,
  identical columns, and numeric agreement to 1e-15 or better. Per-builder results
  are tabulated in `REPO_AUDIT.md`.
- Random seeds are fixed where they affect results: `random_state=42` for PyPREP
  bad-channel detection and for the ICA fit, seed 42 for the balanced subsample
  and for the Gate 1 Test B bootstrap.
- The balanced subsample reproduces exactly, but two details are load bearing.
  The cohort must be passed in its **native row order**, since sorting the pool
  before sampling changes which subjects are drawn, and the seed is applied **per
  sex** rather than once for the whole draw. `build_band_subsample.py` does both
  correctly; anything reimplementing it must match.
- Gate 1 Test B uses `16b_isc_samesex_broadband_gram.py`, the 50-draw Gram-matrix
  implementation. An earlier 20-draw variant existed and is deliberately excluded;
  see `REPO_AUDIT.md` for the evidence identifying 16b as the one that produced
  the published values.

## Development note

The analysis and preprocessing code in this repository was developed with AI
assistance (Anthropic's Claude) under the direction of the author, who is
responsible for the design, the analyses, and their interpretation. This matches
the AI-use disclosure in the paper's acknowledgments. All reported results
regenerate from this code; the subsampling procedures use a fixed random seed (42)
and reproduce the exact subsamples used in the paper.

## License

There is **no LICENSE file in this repository**. One should be added before public
release. A permissive license such as MIT is the usual choice for research code of
this kind, but the decision rests with the author and the institution, so no
license file has been created here.

## Citation

Citation details to be added on acceptance.
