"""Gate 2 Tier 2 STEP 4: leave-one-out ISC under IC-removal conditions A, B, C
and the ocular-only contribution OC = B minus A. Resumable and time-boxed so it
survives an environment that reaps long background jobs.

For each cached subject (scripts/18), regenerate raw_analysis per movie cheaply
from the cached bad channels (no ICA re-fit), resample to 200 Hz, then apply the
cached ICA with three exclusion lists:
  A  exclude all REJECT_LABELS ICs (muscle, eye blink, heart, line, channel)  [published]
  B  exclude REJECT_LABELS MINUS eye blink (ocular retained)
  C  exclude nothing
  OC = B minus A per channel per timepoint (the isolated eye-blink contribution)

Note on OC. The task text wrote "OC = C minus B", but with the conditions above
that expression equals the NON-ocular reject components and is zero when a subject
has no non-ocular reject ICs, contradicting its stated meaning "ocular-only
contribution, the signal that retaining-vs-removing ocular adds back". Retaining
vs removing ocular is the B vs A contrast, so the isolated ocular signal is B
minus A (exactly the eye-blink ICs). We use OC = B minus A and record the slip.
OC is zero (ISC undefined) for a subject with no eye-blink IC; such subjects are
dropped from OC statistics.

Each of A, B, C is re-CAR'd after apply, exactly as the original pipeline. Each
condition is filtered to delta (1 to 4 Hz) and theta (4 to 8 Hz) with the same
FIR firwin zero-phase filter as scripts 13 and 17, then leave-one-out Pearson ISC
is computed per channel per band per condition with the same template
(sum_all minus x_i)/(N minus 1) as script 13.

Resumable design (run after the full subject set is cached, so the present set
and per-movie T_min are stable):
  Phase 4a  cache pre-apply raw_analysis at 200 Hz per subject-movie (skip cached)
  inventory per movie -> present subjects and T_min (cached json)
  sum phase per movie -> incremental sum_all over subjects (checkpointed npz)
  loo  phase per movie -> per-subject LOO rows appended (checkpointed csv)
  assemble -> outputs/gate2_tier2_isc_per_subject.csv once every movie is done

Use --max-minutes to bound a run; relaunch to continue. Overwrites nothing
outside its own cache and the final result CSV.
"""
from __future__ import annotations
import argparse
import csv
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import mne

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
g2 = importlib.import_module("18_gate2_refit")

OUT = PROJECT_ROOT / "outputs"
CACHE_DIR = g2.CACHE_DIR
A200_DIR = CACHE_DIR / "analysis200"
ISC_DIR = CACHE_DIR / "isc"
MOVIES = g2.MOVIES
MIN_DUR_S = g2.MIN_DUR_S
ICA_SFREQ = g2.ICA_SFREQ
FRONTOCENTRAL = g2.FRONTOCENTRAL
PERIOCULAR = g2.PERIOCULAR
BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0)}
CONDS = ["A", "B", "C", "OC"]
KEYS = [f"{c}_{b}" for c in CONDS for b in BANDS]
RESULT_CSV = OUT / "gate2_tier2_isc_per_subject.csv"


def a200_path(sid, task):
    return A200_DIR / f"{sid}_{task}_a200_raw.fif"


def inv_path(task):
    return ISC_DIR / f"inv_{task}.json"


def sum_path(task):
    return ISC_DIR / f"sum_{task}.npz"


def sumprog_path(task):
    return ISC_DIR / f"sumprog_{task}.json"


def rows_path(task):
    return ISC_DIR / f"rows_{task}.csv"


def safe_nanmean(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return float(x.mean()) if x.size else float("nan")


def filter_to_band(data_f32, sfreq, lo, hi):
    return mne.filter.filter_data(
        data_f32.astype(np.float64), sfreq, lo, hi, verbose="ERROR",
        method="fir", fir_design="firwin", phase="zero",
    ).astype(np.float32)


def load_meta(sid):
    with g2.json_path(sid).open() as f:
        return json.load(f)


def ensure_a200(sid, meta, deadline):
    A200_DIR.mkdir(parents=True, exist_ok=True)
    for task in meta["movies_used"]:
        if deadline and time.time() > deadline:
            return False
        outp = a200_path(sid, task)
        if outp.exists():
            continue
        ra = g2.prepare_analysis_cached(sid, task, meta["bads_by_movie"][task])
        if ra is None:
            continue
        if float(ra.info["sfreq"]) > ICA_SFREQ:
            ra.resample(sfreq=ICA_SFREQ, verbose="ERROR")
        g2.retry(lambda: ra.save(str(outp), overwrite=True, verbose="ERROR"),
                 what=f"a200 {sid} {task}")
    return True


def condition_band_data(sid, task, ica, meta, t_min):
    """dict key -> float32 (C, t_min) for one subject-movie."""
    ra = mne.io.read_raw_fif(str(a200_path(sid, task)), preload=True, verbose="ERROR")
    sfreq = float(ra.info["sfreq"])
    ch_names = list(ra.ch_names)
    excl = {"A": meta["reject_idx"], "B": meta["rejectB_idx"], "C": []}
    cond = {}
    for c in ["A", "B", "C"]:
        ica2 = ica.copy()
        ica2.exclude = list(excl[c])
        clean = ica2.apply(ra.copy(), verbose="ERROR")
        clean.set_eeg_reference("average", projection=False, verbose="ERROR")
        cond[c] = clean.get_data().astype(np.float32)[:, :t_min]
        del clean
    cond["OC"] = (cond["B"] - cond["A"]).astype(np.float32)
    del ra
    out = {}
    for c in CONDS:
        for b, (lo, hi) in BANDS.items():
            out[f"{c}_{b}"] = filter_to_band(cond[c], sfreq, lo, hi)
    return out, ch_names


# --------------------------------------------------------------- inventory

def inventory(task, usable, metas):
    if inv_path(task).exists():
        with inv_path(task).open() as f:
            return json.load(f)
    present, t_min, ch_names = [], None, None
    for sid in usable:
        if task not in metas[sid]["movies_used"]:
            continue
        p = a200_path(sid, task)
        if not p.exists():
            return None  # a200 not ready yet
        info = mne.io.read_raw_fif(str(p), preload=False, verbose="ERROR")
        nT = info.n_times
        if (nT / float(info.info["sfreq"])) < MIN_DUR_S.get(task, 0.0):
            continue
        if ch_names is None:
            ch_names = list(info.ch_names)
        t_min = nT if t_min is None else min(t_min, nT)
        present.append(sid)
    inv = {"task": task, "present": present, "t_min": int(t_min),
           "ch_names": ch_names, "C": len(ch_names)}
    ISC_DIR.mkdir(parents=True, exist_ok=True)
    inv_path(task).write_text(json.dumps(inv))
    return inv


# --------------------------------------------------------------- sum phase

def sum_phase(task, inv, metas, icas, deadline):
    """Incrementally accumulate sum_all over present subjects. Returns True if
    the movie sum is complete."""
    present = inv["present"]
    C, t_min = inv["C"], inv["t_min"]
    if sumprog_path(task).exists():
        done = set(json.loads(sumprog_path(task).read_text())["summed"])
        z = np.load(sum_path(task))
        sum_all = {k: z[k].astype(np.float64) for k in KEYS}
    else:
        done = set()
        sum_all = {k: np.zeros((C, t_min), dtype=np.float64) for k in KEYS}
    todo = [s for s in present if s not in done]
    if not todo:
        return True
    print(f"  [sum {task}] {len(done)}/{len(present)} done, summing {len(todo)} more",
          flush=True)
    for n, sid in enumerate(todo):
        if deadline and time.time() > deadline:
            break
        data, _ = condition_band_data(sid, task, icas[sid], metas[sid], t_min)
        for k in KEYS:
            sum_all[k] += data[k]
        done.add(sid)
        del data
        if (n + 1) % 10 == 0 or (n + 1) == len(todo):
            g2.retry(lambda: np.savez(str(sum_path(task)),
                     **{k: sum_all[k].astype(np.float32) for k in KEYS}),
                     what=f"sum save {task}")
            sumprog_path(task).write_text(json.dumps({"summed": sorted(done)}))
            print(f"    [sum {task}] checkpoint {len(done)}/{len(present)}", flush=True)
    return len(done) == len(present)


# --------------------------------------------------------------- loo phase

def loo_phase(task, inv, metas, icas, deadline):
    """Per-subject LOO using the completed sum_all. Append rows; resumable."""
    present = inv["present"]
    C, t_min = inv["C"], inv["t_min"]
    ch_names = inv["ch_names"]
    N = len(present)
    fc_idx = [ch_names.index(c) for c in FRONTOCENTRAL if c in ch_names]
    peri_idx = [ch_names.index(c) for c in PERIOCULAR if c in ch_names]
    z = np.load(sum_path(task))
    sum_all = {k: z[k].astype(np.float64) for k in KEYS}
    inv_n = 1.0 / (N - 1)

    fields = ["participant_id", "movie"] + \
             [f"{c}_{b}_{r}" for c in CONDS for b in BANDS for r in ["fc", "peri", "grand"]]
    done = set()
    if rows_path(task).exists():
        with rows_path(task).open() as f:
            for row in csv.DictReader(f):
                done.add(row["participant_id"])
    todo = [s for s in present if s not in done]
    if not todo:
        return True
    print(f"  [loo {task}] {len(done)}/{N} done, computing {len(todo)} more", flush=True)
    write_header = not rows_path(task).exists()
    fh = rows_path(task).open("a", newline="")
    w = csv.DictWriter(fh, fieldnames=fields)
    if write_header:
        w.writeheader()
    for n, sid in enumerate(todo):
        if deadline and time.time() > deadline:
            break
        data, _ = condition_band_data(sid, task, icas[sid], metas[sid], t_min)
        row = {"participant_id": sid, "movie": task}
        for k in KEYS:
            xb = data[k]
            loo = (sum_all[k] - xb) * inv_n
            a = xb.astype(np.float64)
            bv = loo
            a_c = a - a.mean(axis=1, keepdims=True)
            b_c = bv - bv.mean(axis=1, keepdims=True)
            num = (a_c * b_c).sum(axis=1)
            den = np.sqrt((a_c ** 2).sum(axis=1) * (b_c ** 2).sum(axis=1))
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(den > 0, num / den, np.nan)
            row[f"{k}_fc"] = safe_nanmean(r[fc_idx])
            row[f"{k}_peri"] = safe_nanmean(r[peri_idx])
            row[f"{k}_grand"] = safe_nanmean(r)
            del xb, loo, a, bv, a_c, b_c, num, den, r
        w.writerow(row)
        fh.flush()
        done.add(sid)
        del data
        if (n + 1) % 10 == 0:
            print(f"    [loo {task}] {len(done)}/{N}", flush=True)
    fh.close()
    return len(done) == N


# --------------------------------------------------------------- assemble

def assemble(usable, sexmap):
    frames = []
    for task in MOVIES:
        if rows_path(task).exists():
            frames.append(pd.read_csv(rows_path(task)))
    if not frames:
        return False
    df = pd.concat(frames, ignore_index=True)
    df["sex"] = df["participant_id"].map(sexmap)
    cols = ["participant_id", "sex", "movie"] + \
           [f"{c}_{b}_{r}" for c in CONDS for b in BANDS for r in ["fc", "peri", "grand"]]
    df[cols].to_csv(RESULT_CSV, index=False)
    print(f"Wrote {RESULT_CSV} ({len(df)} subject-movie rows)", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=float, default=None)
    args = ap.parse_args()
    deadline = time.time() + args.max_minutes * 60 if args.max_minutes else None

    bal = g2.build_balanced_150()
    sexmap = dict(zip(bal.participant_id, bal.sex))
    usable = [s for s in bal.participant_id if g2.cache_complete(s)]
    nM = sum(1 for s in usable if sexmap[s] == "M")
    nF = sum(1 for s in usable if sexmap[s] == "F")
    print(f"=== STEP 4 (resumable). cached subjects: {len(usable)} (M={nM} F={nF}) ===",
          flush=True)
    ISC_DIR.mkdir(parents=True, exist_ok=True)
    metas = {s: load_meta(s) for s in usable}

    # Phase 4a: ensure analysis200 cache (time-boxed)
    print("  Phase 4a: cache raw_analysis at 200 Hz", flush=True)
    for sid in usable:
        if not ensure_a200(sid, metas[sid], deadline):
            print("  TIME BUDGET during Phase 4a; relaunch to continue", flush=True)
            return
    print("  Phase 4a complete", flush=True)

    icas = {s: mne.preprocessing.read_ica(str(g2.ica_path(s)), verbose="ERROR")
            for s in usable}

    all_done = True
    for task in MOVIES:
        inv = inventory(task, usable, metas)
        if inv is None:
            print(f"  {task}: a200 not ready for all subjects; skipping", flush=True)
            all_done = False
            continue
        if not sum_phase(task, inv, metas, icas, deadline):
            print(f"  {task}: sum phase incomplete (budget); relaunch", flush=True)
            all_done = False
            continue
        if not loo_phase(task, inv, metas, icas, deadline):
            print(f"  {task}: loo phase incomplete (budget); relaunch", flush=True)
            all_done = False
            continue
        print(f"  {task}: DONE", flush=True)

    if all_done:
        assemble(usable, sexmap)
        print("STEP 4 COMPLETE", flush=True)
    else:
        print("STEP 4 PARTIAL: relaunch to continue", flush=True)


if __name__ == "__main__":
    main()
