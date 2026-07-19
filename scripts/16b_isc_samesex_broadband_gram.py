"""Broadband full-cohort same-sex, size-matched ISC (Test B), Gram-matrix method.

Same analysis as scripts/16_isc_samesex_broadband.py (the literal 2:1 case), but
reformulated so the male size-matched bootstrap is free of per-draw disk I/O.

The leave-one-out per-channel Pearson r between subject i and the mean of a
subject set S (i in S) depends only on inner products of the time-mean-centered
signals. With the per-channel Gram matrix
    G[c][i, j] = <xc_i[c], xc_j[c]>            (xc = x minus its time mean)
the ISC of subject i against the leave-one-out mean of any set S is, per channel,
    rowsum_i = sum_{j in S} G[c][i, j]
    totalS   = sum_{j,k in S} G[c][j, k]
    num      = rowsum_i - G[c][i, i]
    var_T    = totalS - 2*rowsum_i + G[c][i, i]
    r        = num / ( sqrt(G[c][i, i]) * sqrt(var_T) )
which is exactly corr(x_i, (sum_S - x_i)/(|S|-1)) since correlation is scale
invariant. So the Gram is built once per movie (each subject read once), then the
pooled ISC, the female same-sex ISC, and every male bootstrap draw are closed
form arithmetic in RAM. This reproduces script 11 to floating point (the only
difference vs 11 is that 11 accumulated sum_all in float32; the Gram is float64,
so agreement is at the 1e-8 level, well inside the 1e-6 gate).

Design (identical statistics to script 16):
  - each female against (sum_F - x) / (N_F - 1)         [all other females]
  - each male against the leave-one-out mean of a random subsample of the males
    down to the female n, averaged over B draws (seed 42); the female template
    and the size-matched male template both aggregate (N_F - 1) subjects.

Robustness: 8 GB RAM forces a per-movie float32 memmap on the SSD; the Gram is
built by reading one channel at a time so peak RAM stays bounded. The USB SSD can
transiently unmount, so reads are retried, the run waits for remount, and each
finished movie is checkpointed to LOCAL disk for resume. Launch under caffeinate.

Outputs (NEW files, originals untouched):
  outputs/repro_R12345678_isc_per_subject.csv                 [pooled, repro]
  outputs/R12345678_isc_per_subject_samesex_sizematched.csv   [Test B]
  outputs/R12345678_testB_draw_sd.csv                         [across-draw SD]
"""
import csv
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import mne

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COHORT_CSV = PROJECT_ROOT / "outputs" / "R12345678_isc_cohort.csv"
ORIG_PER_SUBJECT = PROJECT_ROOT / "outputs" / "R12345678_isc_per_subject.csv"
OUT_REPRO = PROJECT_ROOT / "outputs" / "repro_R12345678_isc_per_subject.csv"
OUT_SS = PROJECT_ROOT / "outputs" / "R12345678_isc_per_subject_samesex_sizematched.csv"
OUT_SD = PROJECT_ROOT / "outputs" / "R12345678_testB_draw_sd.csv"
CKPT_DIR = PROJECT_ROOT / "outputs" / "_testB_ckpt"
# Scratch location for the per-movie float32 memmap. This is large (tens of GB
# per movie at full cohort size) and is deleted after each movie completes, so
# point it at a fast disk with room to spare. Override with the environment
# variable HBN_SCRATCH_ROOT; defaults to ./scratch under the project root.
# The original analysis used an external SSD at
# /Volumes/PortableSSD/Projects/HBN-BrainAI, hence the remount handling below.
SSD_ROOT = Path(os.environ.get("HBN_SCRATCH_ROOT", str(PROJECT_ROOT / "scratch")))
MEMMAP_DIR = SSD_ROOT / "tmp_testB"

MOVIES = ["DespicableMe", "DiaryOfAWimpyKid", "FunwithFractals", "ThePresent"]
MIN_DUR_S = {"ThePresent": 200.0}  # match the script 12 ThePresent patch
B_DRAWS = 50
GATE_TOL = 1e-6


def fif_path(release, sid, movie):
    return PROJECT_ROOT / "outputs" / f"preprocessed_movies_{release}" / \
        f"{sid}_{movie}_preproc_v3concat_raw.fif"


def ssd_available():
    """True when the scratch volume is present.

    The original run kept scratch on a removable USB SSD that could transiently
    unmount mid-run, which is what the retry and wait logic below exists for. On
    a fixed local disk this simply returns True once the directory exists.
    """
    try:
        SSD_ROOT.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    return SSD_ROOT.exists()


def wait_for_ssd(timeout=900, poll=15):
    waited = 0
    while not ssd_available():
        if waited >= timeout:
            raise RuntimeError(f"SSD did not remount within {timeout}s")
        time.sleep(poll)
        waited += poll
        print(f"    waiting for SSD remount... {waited}s", flush=True)


def read_fif_data(rel, sid, movie, t_min, attempts=6):
    last = None
    for k in range(attempts):
        try:
            raw = mne.io.read_raw_fif(str(fif_path(rel, sid, movie)),
                                       preload=True, verbose="ERROR")
            x = raw.get_data().astype(np.float32)[:, :t_min]
            del raw
            return x
        except OSError as e:
            last = e
            print(f"    OSError reading {sid}/{movie}: {e}; waiting for SSD...", flush=True)
            wait_for_ssd()
            time.sleep(2)
    raise last


def pass1_inventory(cohort_df, movie, attempts=6):
    for k in range(attempts):
        try:
            accepted = []
            ch_names_ref, sfreq_ref, t_min, skipped = None, None, None, 0
            min_dur = MIN_DUR_S.get(movie, 0.0)
            for _, row in cohort_df.iterrows():
                sid, rel, sex = row["participant_id"], row["release"], row["sex"]
                fif = fif_path(rel, sid, movie)
                if not fif.exists():
                    if not ssd_available():
                        raise OSError("SSD vanished during inventory")
                    skipped += 1
                    continue
                raw = mne.io.read_raw_fif(str(fif), preload=False, verbose="ERROR")
                names = list(raw.ch_names)
                sf = float(raw.info["sfreq"])
                nT = raw.n_times
                if (nT / sf) < min_dur:
                    skipped += 1
                    continue
                if ch_names_ref is None:
                    ch_names_ref, sfreq_ref, t_min = names, sf, nT
                else:
                    if names != ch_names_ref or sf != sfreq_ref:
                        skipped += 1
                        continue
                    if nT < t_min:
                        t_min = nT
                accepted.append((sid, rel, sex))
            return accepted, ch_names_ref, sfreq_ref, t_min, skipped
        except OSError as e:
            print(f"    OSError during inventory of {movie}: {e}; waiting for SSD...", flush=True)
            wait_for_ssd()
            time.sleep(2)
    raise OSError(f"inventory failed for {movie}")


def loo_isc_full(G):
    """Pooled leave-one-out ISC for every subject. G: (C, N, N). Returns (N, C)."""
    rowsum = G.sum(axis=2)                              # (C, N)
    total = G.sum(axis=(1, 2))                          # (C,)
    Gii = np.diagonal(G, axis1=1, axis2=2)              # (C, N)
    num = rowsum - Gii
    var_T = total[:, None] - 2.0 * rowsum + Gii
    return _finish(num, Gii, var_T).T                  # (N, C)


def loo_isc_subset(G, subset):
    """Leave-one-out ISC for each member of subset, template = the subset.
    G: (C, N, N). subset: 1D index array. Returns (len(subset), C)."""
    Gs = G[:, subset][:, :, subset]                    # (C, k, k)
    rowsum = Gs.sum(axis=2)                            # (C, k)
    total = Gs.sum(axis=(1, 2))                        # (C,)
    Gii = np.diagonal(Gs, axis1=1, axis2=2)           # (C, k)
    num = rowsum - Gii
    var_T = total[:, None] - 2.0 * rowsum + Gii
    return _finish(num, Gii, var_T).T                 # (k, C)


def _finish(num, Gii, var_T):
    var_T = np.where(var_T > 0, var_T, np.nan)
    den = np.sqrt(Gii) * np.sqrt(var_T)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where((den > 0) & (Gii > 0), num / den, np.nan)
    return r


def build_gram(mm, N, C):
    """Per-channel Gram of time-mean-centered signals. Returns (C, N, N) float64.
    Reads the memmap one channel at a time to bound peak RAM."""
    G = np.empty((C, N, N), dtype=np.float64)
    ts = time.time()
    for c in range(C):
        Xc = np.array(mm[:, c, :], dtype=np.float64)   # (N, t_min)
        Xc -= Xc.mean(axis=1, keepdims=True)
        G[c] = Xc @ Xc.T
        del Xc
        if (c + 1) % 32 == 0:
            print(f"    gram channel {c+1}/{C} ({time.time()-ts:.1f}s)", flush=True)
    print(f"  gram built in {time.time()-ts:.1f}s ({G.nbytes/1e9:.2f} GB)", flush=True)
    return G


def gate_movie(movie, accepted, ch_names, isc_pool):
    orig = pd.read_csv(ORIG_PER_SUBJECT)
    orig = orig[orig.movie == movie].set_index("participant_id")
    grand = np.nanmean(isc_pool, axis=1)
    max_abs, n_unmatched = 0.0, 0
    for i, (sid, rel, sex) in enumerate(accepted):
        if sid not in orig.index:
            n_unmatched += 1
            continue
        orow = orig.loc[sid]
        ov = pd.to_numeric(orow["isc_grand_mean"], errors="coerce")
        rv = grand[i]
        if not (np.isnan(ov) and np.isnan(rv)):
            max_abs = max(max_abs, abs(float(ov) - float(rv)))
        for j, ch in enumerate(ch_names):
            ov = pd.to_numeric(orow[f"isc_{ch}"], errors="coerce")
            rv = isc_pool[i, j]
            on, rn = np.isnan(ov), np.isnan(rv)
            if on and rn:
                continue
            if on ^ rn:
                max_abs = np.inf
                continue
            max_abs = max(max_abs, abs(float(ov) - float(rv)))
    return max_abs, n_unmatched


def process_movie(movie, cohort, rng):
    accepted, ch_names, sfreq, t_min, skipped = pass1_inventory(cohort, movie)
    N, C = len(accepted), len(ch_names)
    sexes = np.array([s for (_, _, s) in accepted])
    male_idx = np.where(sexes == "M")[0]
    female_idx = np.where(sexes == "F")[0]
    N_M, N_F = len(male_idx), len(female_idx)
    print(f"  pass1: N={N} (N_M={N_M}, N_F={N_F}; skipped {skipped}); "
          f"sfreq={sfreq}, T_min={t_min} ({t_min/sfreq:.1f}s)", flush=True)

    MEMMAP_DIR.mkdir(parents=True, exist_ok=True)
    mm_path = MEMMAP_DIR / f"{movie}_data.f32"
    mm = np.memmap(str(mm_path), dtype=np.float32, mode="w+", shape=(N, C, t_min))
    ts = time.time()
    for i, (sid, rel, sex) in enumerate(accepted):
        mm[i] = read_fif_data(rel, sid, movie, t_min)
    mm.flush()
    print(f"  memmap in {time.time()-ts:.1f}s ({mm_path.stat().st_size/1e9:.1f} GB)", flush=True)

    G = build_gram(mm, N, C)
    del mm
    try:
        os.remove(mm_path)
    except OSError:
        pass

    # pooled reproduction
    isc_pool = loo_isc_full(G)
    max_abs, n_unmatched = gate_movie(movie, accepted, ch_names, isc_pool)
    status = ("OK" if max_abs < 1e-9 else
              ("WITHIN FP (proceed)" if max_abs < GATE_TOL else "FAIL"))
    print(f"  GATE {movie}: max abs diff={max_abs:.3e} unmatched={n_unmatched} -> {status}", flush=True)
    if not (max_abs < GATE_TOL):
        raise SystemExit(f"STOP: broadband reproduction for {movie} exceeds {GATE_TOL}.")

    # female same-sex (deterministic)
    isc_ss = np.full((N, C), np.nan, dtype=np.float64)
    isc_ss[female_idx, :] = loo_isc_subset(G, female_idx)

    # male size-matched bootstrap
    K = N_F
    male_sum = np.zeros((N, C), dtype=np.float64)
    male_cnt = np.zeros(N, dtype=np.int64)
    draw_group_means = []
    ts = time.time()
    for b in range(B_DRAWS):
        subset = np.sort(rng.choice(male_idx, size=K, replace=False))
        risc = loo_isc_subset(G, subset)               # (K, C)
        male_sum[subset, :] += risc
        male_cnt[subset] += 1
        draw_group_means.append(float(np.nanmean(np.nanmean(risc, axis=1))))
    for j in male_idx:
        if male_cnt[j] > 0:
            isc_ss[j, :] = male_sum[j] / male_cnt[j]
    dg = np.array(draw_group_means)
    print(f"  male bootstrap ({B_DRAWS} draws) in {time.time()-ts:.1f}s; "
          f"across-draw group mean={dg.mean():.5f} SD={dg.std(ddof=1):.5f}", flush=True)
    del G

    grand_p = np.nanmean(isc_pool, axis=1)
    grand_s = np.nanmean(isc_ss, axis=1)
    repro_rows, ss_rows = [], []
    for i, (sid, rel, sex) in enumerate(accepted):
        rp = {"participant_id": sid, "release": rel, "movie": movie,
              "isc_grand_mean": float(grand_p[i])}
        rs = {"participant_id": sid, "release": rel, "movie": movie, "sex": sex,
              "isc_grand_mean": float(grand_s[i]) if not np.isnan(grand_s[i]) else "",
              "n_draws": int(male_cnt[i]) if sex == "M" else "deterministic"}
        for j, ch in enumerate(ch_names):
            vp, vs = isc_pool[i, j], isc_ss[i, j]
            rp[f"isc_{ch}"] = float(vp) if not np.isnan(vp) else ""
            rs[f"isc_{ch}"] = float(vs) if not np.isnan(vs) else ""
        repro_rows.append(rp)
        ss_rows.append(rs)
    sd_row = {"movie": movie, "N_M": N_M, "N_F": N_F, "K_matched": K, "B_draws": B_DRAWS,
              "draw_group_mean": float(dg.mean()), "draw_group_sd": float(dg.std(ddof=1))}
    return ch_names, repro_rows, ss_rows, sd_row


def main():
    cohort = pd.read_csv(COHORT_CSV)
    print("=== Test B (Gram method): broadband full-cohort same-sex size-matched ===", flush=True)
    print(f"  cohort n={len(cohort)} sex={cohort.sex.value_counts().to_dict()}", flush=True)
    print(f"  B_DRAWS={B_DRAWS}, seed=42, gate tol={GATE_TOL}", flush=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    rng = np.random.default_rng(42)

    ch_names_ref = None
    for movie in MOVIES:
        ckpt = CKPT_DIR / f"{movie}.pkl"
        if ckpt.exists():
            print(f"\n--- {movie} (resume) ---", flush=True)
            with ckpt.open("rb") as f:
                data = pickle.load(f)
            n_m, k = data["sd_row"]["N_M"], data["sd_row"]["K_matched"]
            for _ in range(B_DRAWS):
                rng.choice(np.arange(n_m), size=k, replace=False)
            ch_names_ref = data["ch_names"]
            continue
        print(f"\n--- {movie} ---", flush=True)
        attempt = 0
        while True:
            attempt += 1
            try:
                ch_names, repro_rows, ss_rows, sd_row = process_movie(movie, cohort, rng)
                break
            except OSError as e:
                print(f"  movie {movie} attempt {attempt} OSError: {e}", flush=True)
                try:
                    os.remove(MEMMAP_DIR / f"{movie}_data.f32")
                except OSError:
                    pass
                wait_for_ssd()
                if attempt >= 4:
                    raise
                rng = np.random.default_rng(42 + 1000 * attempt)
        ch_names_ref = ch_names
        with ckpt.open("wb") as f:
            pickle.dump({"ch_names": ch_names, "repro_rows": repro_rows,
                         "ss_rows": ss_rows, "sd_row": sd_row}, f)
        print(f"  checkpointed {movie}", flush=True)

    repro_rows, ss_rows, sd_rows = [], [], []
    for movie in MOVIES:
        with (CKPT_DIR / f"{movie}.pkl").open("rb") as f:
            data = pickle.load(f)
        repro_rows += data["repro_rows"]
        ss_rows += data["ss_rows"]
        sd_rows.append(data["sd_row"])
        ch_names_ref = data["ch_names"]

    repro_fields = ["participant_id", "release", "movie", "isc_grand_mean"] + \
                   [f"isc_{c}" for c in ch_names_ref]
    ss_fields = ["participant_id", "release", "movie", "sex", "n_draws", "isc_grand_mean"] + \
                [f"isc_{c}" for c in ch_names_ref]
    with OUT_REPRO.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=repro_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(repro_rows)
    print(f"\nWrote {OUT_REPRO} ({len(repro_rows)} rows)", flush=True)
    with OUT_SS.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ss_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(ss_rows)
    print(f"Wrote {OUT_SS} ({len(ss_rows)} rows)", flush=True)
    pd.DataFrame(sd_rows).to_csv(OUT_SD, index=False)
    print(f"Wrote {OUT_SD}", flush=True)

    try:
        MEMMAP_DIR.rmdir()
    except OSError:
        pass
    for movie in MOVIES:
        try:
            os.remove(CKPT_DIR / f"{movie}.pkl")
        except OSError:
            pass
    try:
        CKPT_DIR.rmdir()
    except OSError:
        pass

    print(f"\nTotal wall: {(time.time()-t_start)/60:.1f} min", flush=True)
    print("STEP 7 COMPUTE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
