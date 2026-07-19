"""
Banded ridge TRF for fair LOWLEVEL vs CLIP_PCA comparison.

Differences vs scripts/trf_batch.py:
- LOWLEVEL alone: alpha tuned per subject over a 5-point grid via inner 5-fold CV.
- CLIP_PCA alone: same.
- COMBINED: separate alphas for the LL block (1210 lagged features) and
  CLIP block (6050 lagged features), tuned over a 5x5 grid (25 combos)
  via inner 5-fold CV.

Speed strategy: for each inner-CV training fold, compute X^T X and X^T y
ONCE on the (z-scored, per-block-scaled) features, then for each alpha
combination just modify the diagonal of X^T X and solve the normal
equations. This avoids 25x re-fitting the full ridge for COMBINED. Banded
ridge is implemented by adding a per-block diagonal penalty to X^T X
directly (no feature rescaling needed), so the same X^T X is reused
across all alpha pairs.

Speedup approximation: alpha selection is done ONCE per subject, using
the mean across channels as a single representative target. Outer 5-fold
CV per channel uses those fixed alphas. Documented as a known
approximation.

Outputs: outputs/trf_banded_30subj_results.csv with one row per
(subject, model, channel) and the chosen alpha(s) recorded.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SUBJECT_LIST = "data/trf_pilot_30_subjects.txt"
PREPROC_DIR = "outputs/preprocessed_movies_R1"
LL_CSV = "outputs/movie_features_lowlevel/ThePresent_lowlevel.csv"
CLIP_CSV = "outputs/movie_features_clip/ThePresent_clip.csv"
OUT_CSV = "outputs/trf_banded_30subj_results.csv"

SFREQ = 200
MAX_LAG_MS = 600
N_LAGS = int(MAX_LAG_MS / 1000 * SFREQ) + 1  # 121
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 5
N_PCA = 50
# Single-block grid (cheap): full 5-point grid
ALPHAS_SINGLE = [10.0, 100.0, 1000.0, 10000.0, 100000.0]
# COMBINED grid (expensive 25-pair version originally): tighter 3-point grid
# spans the same range. Reduces 5x5=25 pairs to 3x3=9. The first 11 subjects
# fit with the full 5x5 grid all converged on alpha=10 for both blocks; this
# tighter grid still includes that endpoint and brackets it.
ALPHAS_COMBINED = [10.0, 1000.0, 100000.0]


def upsample_to_sfreq(df: pd.DataFrame, sfreq: int) -> np.ndarray:
    arr = df.drop(columns=["time_sec"]).values.astype(np.float64)
    return np.repeat(arr, sfreq, axis=0)


def lag_matrix(X: np.ndarray, n_lags: int) -> np.ndarray:
    T, F = X.shape
    out = np.zeros((T, F * n_lags), dtype=np.float32)
    for k in range(n_lags):
        if k == 0:
            out[:, :F] = X
        else:
            out[k:, k * F:(k + 1) * F] = X[:-k]
    return out


def _fold_edges(T: int, n_folds: int) -> np.ndarray:
    return np.linspace(0, T, n_folds + 1, dtype=int)


def _fit_normal_eq(XTX: np.ndarray, XTy: np.ndarray, alpha_diag: np.ndarray) -> np.ndarray:
    """Ridge solution via normal equations. XTX assumed symmetrized + float64."""
    A = XTX + np.diag(alpha_diag)
    try:
        c, low = cho_factor(A, lower=True, overwrite_a=True, check_finite=False)
        return cho_solve((c, low), XTy, overwrite_b=False, check_finite=False)
    except np.linalg.LinAlgError:
        # Fall back to a generic solver with extra jitter for ill-conditioned cases
        A2 = XTX + np.diag(alpha_diag) + np.eye(XTX.shape[0]) * 1e-6
        return np.linalg.solve(A2, XTy)


def _xtx_symmetric_f64(X: np.ndarray) -> np.ndarray:
    """Compute X^T X in float32 (cheap) then symmetrize and upcast to float64.
    Float32 BLAS doesn't guarantee bit-exact symmetry; symmetrizing avoids
    spurious 'not positive definite' errors during Cholesky."""
    XTX_f32 = X.T @ X
    XTX = XTX_f32.astype(np.float64, copy=False)
    XTX = 0.5 * (XTX + XTX.T)
    return XTX


def _zscore_blocks_train(X_blocks: list[np.ndarray]) -> tuple[list, list[np.ndarray]]:
    """Manual per-block z-score in float32 (avoid sklearn StandardScaler's
    float64 promotion which doubles memory on the (T, ~6000) blocks)."""
    out_scalers = []
    out_Z = []
    for Xb in X_blocks:
        mu = Xb.mean(axis=0, dtype=np.float64).astype(np.float32)
        sd = Xb.std(axis=0, dtype=np.float64).astype(np.float32)
        sd[sd < 1e-8] = 1.0
        Z = (Xb - mu) / sd
        out_scalers.append((mu, sd))
        out_Z.append(Z.astype(np.float32, copy=False))
    return out_scalers, out_Z


def _apply_scalers(scalers, X_blocks: list[np.ndarray]) -> list[np.ndarray]:
    return [((Xb - mu) / sd).astype(np.float32, copy=False)
            for (mu, sd), Xb in zip(scalers, X_blocks)]


def _build_block_alpha_diag(block_sizes: list[int], alphas: list[float]) -> np.ndarray:
    diag = np.empty(sum(block_sizes), dtype=np.float64)
    cur = 0
    for sz, a in zip(block_sizes, alphas):
        diag[cur:cur + sz] = a
        cur += sz
    return diag


def select_alphas_via_inner_cv(
    X_blocks: list[np.ndarray], y: np.ndarray, alpha_grid: list[list[float]], n_folds: int
) -> tuple[list[float], list]:
    """Generic inner-CV alpha selection.

    X_blocks: per-block feature matrices (each shape (T, F_b)). Concatenation order = block order.
    y: 1-D target time series (T,).
    alpha_grid: list of lists. The cross product is the search space.
        e.g. for single-block: [[10, 100, 1000]] (3 candidates).
        For two-block:        [[10, 100, ...], [10, 100, ...]] (cross product).
    Returns (best_alpha_list, results_grid).
    Each entry in results_grid is (alpha_tuple, mean_r_across_folds).
    """
    from itertools import product

    T = len(y)
    block_sizes = [Xb.shape[1] for Xb in X_blocks]
    edges = _fold_edges(T, n_folds)
    candidates = list(product(*alpha_grid))

    fold_pred_r: dict[tuple, list[float]] = {a: [] for a in candidates}

    for fi in range(n_folds):
        s, e = edges[fi], edges[fi + 1]
        mask = np.ones(T, dtype=bool); mask[s:e] = False

        scalers, Ztr_blocks = _zscore_blocks_train([Xb[mask] for Xb in X_blocks])
        Zte_blocks = _apply_scalers(scalers, [Xb[s:e] for Xb in X_blocks])
        Xtr = np.concatenate(Ztr_blocks, axis=1)
        Xte = np.concatenate(Zte_blocks, axis=1)
        del Ztr_blocks, Zte_blocks

        ytr, yte = y[mask], y[s:e]
        ymean = float(ytr.mean())
        ytr_c = (ytr - ymean).astype(np.float32, copy=False)

        XTX = _xtx_symmetric_f64(Xtr)
        XTy = (Xtr.T @ ytr_c).astype(np.float64, copy=False)
        del Xtr

        for alphas in candidates:
            diag = _build_block_alpha_diag(block_sizes, list(alphas))
            w = _fit_normal_eq(XTX.copy(), XTy, diag)
            pred = (Xte @ w.astype(np.float32) + ymean)
            if yte.std() < 1e-12 or pred.std() < 1e-12:
                fold_pred_r[alphas].append(0.0)
            else:
                fold_pred_r[alphas].append(pearsonr(yte, pred)[0])
        del XTX, XTy, Xte

    grid_results = [(a, float(np.mean(rs))) for a, rs in fold_pred_r.items()]
    best = max(grid_results, key=lambda t: t[1])
    return list(best[0]), grid_results


def cv_per_channel(
    X_blocks: list[np.ndarray], Y: np.ndarray, n_folds: int, alphas: list[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Outer CV: fit ridge with given per-block alphas, return per-channel
    mean and std of Pearson r across folds. Uses normal equations + Cholesky
    once per fold; multi-output via XTY (P x C)."""
    T, C = Y.shape
    block_sizes = [Xb.shape[1] for Xb in X_blocks]
    edges = _fold_edges(T, n_folds)
    rs = np.zeros((n_folds, C), dtype=np.float64)

    diag = _build_block_alpha_diag(block_sizes, alphas)

    for fi in range(n_folds):
        s, e = edges[fi], edges[fi + 1]
        mask = np.ones(T, dtype=bool); mask[s:e] = False

        scalers, Ztr_blocks = _zscore_blocks_train([Xb[mask] for Xb in X_blocks])
        Zte_blocks = _apply_scalers(scalers, [Xb[s:e] for Xb in X_blocks])
        Xtr = np.concatenate(Ztr_blocks, axis=1)
        Xte = np.concatenate(Zte_blocks, axis=1)
        del Ztr_blocks, Zte_blocks

        Ytr = Y[mask]
        Yte = Y[s:e]
        ymean = Ytr.mean(axis=0)
        Ytr_c = (Ytr - ymean).astype(np.float32, copy=False)

        XTX = _xtx_symmetric_f64(Xtr)
        XTY = (Xtr.T @ Ytr_c).astype(np.float64, copy=False)
        del Xtr, Ytr_c

        A = XTX + np.diag(diag)
        try:
            c, low = cho_factor(A, lower=True, overwrite_a=True, check_finite=False)
            W = cho_solve((c, low), XTY, overwrite_b=False, check_finite=False)
        except np.linalg.LinAlgError:
            A2 = XTX + np.diag(diag) + np.eye(XTX.shape[0]) * 1e-6
            W = np.linalg.solve(A2, XTY)
            c = None
        Yhat = Xte @ W.astype(np.float32) + ymean
        del XTX, XTY, A, c, Xte

        for cc in range(C):
            yt, yp = Yte[:, cc], Yhat[:, cc]
            if yt.std() < 1e-12 or yp.std() < 1e-12:
                rs[fi, cc] = 0.0
            else:
                rs[fi, cc] = pearsonr(yt, yp)[0]

    return rs.mean(axis=0), rs.std(axis=0)


def build_movie_features() -> tuple[np.ndarray, np.ndarray]:
    ll = pd.read_csv(LL_CSV)
    for c in [c for c in ll.columns if c != "time_sec"]:
        if ll[c].isna().any():
            ll[c] = ll[c].fillna(ll[c].mean())
    LL = upsample_to_sfreq(ll, SFREQ)

    clip = pd.read_csv(CLIP_CSV)
    pca = PCA(n_components=N_PCA, random_state=0)
    clip_pcs = pca.fit_transform(clip.drop(columns=["time_sec"]).values)
    clip_pcs_df = pd.DataFrame(clip_pcs, columns=[f"clip_pc{i}" for i in range(N_PCA)])
    clip_pcs_df.insert(0, "time_sec", clip["time_sec"].values)
    CLIP_PCA = upsample_to_sfreq(clip_pcs_df, SFREQ)

    print(f"[movie] LL upsampled {LL.shape}; CLIP PCA cum EV {pca.explained_variance_ratio_.sum():.4f}; CLIP_PCA upsampled {CLIP_PCA.shape}", flush=True)

    LL_lag = lag_matrix(LL, N_LAGS)
    CLIP_lag = lag_matrix(CLIP_PCA, N_LAGS)
    print(f"[movie] lagged LL {LL_lag.shape}, CLIP {CLIP_lag.shape}", flush=True)
    return LL_lag, CLIP_lag


def fit_subject(subject: str, LL_lag_full: np.ndarray, CLIP_lag_full: np.ndarray) -> tuple[pd.DataFrame, dict]:
    fif = Path(PREPROC_DIR) / f"{subject}_ThePresent_preproc_v3concat_raw.fif"
    raw = mne.io.read_raw_fif(fif, preload=True, verbose="ERROR")
    eeg = raw.get_data().T.astype(np.float32, copy=False)  # (T, C)
    n_use = min(eeg.shape[0], LL_lag_full.shape[0])
    eeg = eeg[:n_use]
    LL_lag = LL_lag_full[:n_use]
    CLIP_lag = CLIP_lag_full[:n_use]

    y_rep = eeg.mean(axis=1).astype(np.float64, copy=False)

    t_a = time.time()
    a_ll_list, _ = select_alphas_via_inner_cv([LL_lag], y_rep, [ALPHAS_SINGLE], N_INNER_FOLDS)
    a_clip_list, _ = select_alphas_via_inner_cv([CLIP_lag], y_rep, [ALPHAS_SINGLE], N_INNER_FOLDS)
    pair_list, _ = select_alphas_via_inner_cv([LL_lag, CLIP_lag], y_rep, [ALPHAS_COMBINED, ALPHAS_COMBINED], N_INNER_FOLDS)
    t_alpha = time.time() - t_a

    a_ll = a_ll_list[0]
    a_clip = a_clip_list[0]
    pair_ll, pair_clip = pair_list[0], pair_list[1]

    t_o = time.time()
    rs_ll_m, rs_ll_s = cv_per_channel([LL_lag], eeg, N_OUTER_FOLDS, [a_ll])
    rs_cl_m, rs_cl_s = cv_per_channel([CLIP_lag], eeg, N_OUTER_FOLDS, [a_clip])
    rs_co_m, rs_co_s = cv_per_channel([LL_lag, CLIP_lag], eeg, N_OUTER_FOLDS, [pair_ll, pair_clip])
    t_outer = time.time() - t_o

    rows = []
    for c, ch in enumerate(raw.ch_names):
        rows.append({"participant_id": subject, "model": "LOWLEVEL", "channel": ch,
                     "r_cv_mean": float(rs_ll_m[c]), "r_cv_std": float(rs_ll_s[c]),
                     "alpha_ll": a_ll, "alpha_clip": np.nan})
        rows.append({"participant_id": subject, "model": "CLIP_PCA", "channel": ch,
                     "r_cv_mean": float(rs_cl_m[c]), "r_cv_std": float(rs_cl_s[c]),
                     "alpha_ll": np.nan, "alpha_clip": a_clip})
        rows.append({"participant_id": subject, "model": "COMBINED", "channel": ch,
                     "r_cv_mean": float(rs_co_m[c]), "r_cv_std": float(rs_co_s[c]),
                     "alpha_ll": pair_ll, "alpha_clip": pair_clip})

    info = {"a_ll": a_ll, "a_clip": a_clip, "pair_ll": pair_ll, "pair_clip": pair_clip,
            "t_alpha": t_alpha, "t_outer": t_outer}
    return pd.DataFrame(rows), info


def main():
    t_start = time.time()
    with open(SUBJECT_LIST) as f:
        subjects = [s.strip() for s in f if s.strip()]
    print(f"Subjects: {len(subjects)}", flush=True)

    LL_lag_full, CLIP_lag_full = build_movie_features()

    out_path = Path(OUT_CSV)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows, fail = [], []
    done_subjects: set[str] = set()
    if out_path.exists():
        prev = pd.read_csv(out_path)
        done_subjects = set(prev["participant_id"].unique())
        all_rows.append(prev)
        print(f"Resuming: {len(done_subjects)} subjects already in {out_path}: {sorted(done_subjects)[:3]}...", flush=True)

    for i, s in enumerate(subjects, 1):
        if s in done_subjects:
            print(f"[{i:02d}/{len(subjects)}] {s}: SKIP (already done)", flush=True)
            continue
        ts = time.time()
        try:
            df, info = fit_subject(s, LL_lag_full, CLIP_lag_full)
            all_rows.append(df)
            med = df.groupby("model")["r_cv_mean"].median().to_dict()
            print(
                f"[{i:02d}/{len(subjects)}] {s}: ok in {time.time()-ts:.1f}s "
                f"(alpha {info['t_alpha']:.1f}s + outer {info['t_outer']:.1f}s) | "
                f"a_ll={info['a_ll']:g} a_clip={info['a_clip']:g} "
                f"pair=({info['pair_ll']:g},{info['pair_clip']:g}) | medians {med}",
                flush=True,
            )
        except Exception as e:
            fail.append((s, str(e)))
            print(f"[{i:02d}/{len(subjects)}] {s}: FAILED in {time.time()-ts:.1f}s: {e}", flush=True)
            traceback.print_exc()
            continue

        if i % 5 == 0:
            pd.concat(all_rows).to_csv(out_path, index=False)
            print(f"  checkpoint: {len(all_rows)} subjects so far", flush=True)

    if all_rows:
        pd.concat(all_rows).to_csv(out_path, index=False)
        print(f"Wrote {out_path} ({len(all_rows)} subjects, {len(fail)} failed)", flush=True)
    if fail:
        print(f"Failures ({len(fail)}):")
        for s, e in fail:
            print(f"  {s}: {e}")
    print(f"Total wall: {time.time() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
