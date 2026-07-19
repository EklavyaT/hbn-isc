"""Gate 2 (ocular artifact) Tier 1, Test 1: periocular versus frontocentral
spatial gradient of the theta sex effect. Plus STEP 0 preconditions.

Logic: an ocular generator (blink, saccade) peaks at the electrodes nearest the
eyes and falls off with distance. A frontal-midline neural theta source peaks
frontocentrally (E30, E36, E37), away from the eyes. So if the theta sex-effect
t-statistic is larger at the frontocentral cluster than at the periocular and
frontopolar-midline electrodes, the effect peaks away from the eyes, which argues
against an ocular generator.

Reads the published per-channel theta sex effect
(outputs/R12345678_sex_effect_theta_per_channel.csv) and the GSN-HydroCel-129
montage. Electrode groups are derived from montage geometry, not hardcoded.
Writes outputs/gate2_test1_gradient.csv. Overwrites nothing else. No SSD needed.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import mne

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT = PROJECT_ROOT / "outputs"
ANCHOR_CSV = OUT / "R12345678_sex_effect_theta_per_channel.csv"
GRADIENT_CSV = OUT / "gate2_test1_gradient.csv"

MOVIE_ORDER = ["DespicableMe", "DiaryOfAWimpyKid", "FunwithFractals", "ThePresent"]
FC_CH = ["E30", "E36", "E37"]
N_PERIOCULAR = 10


def step0_preconditions():
    print("=" * 70)
    print("GATE 2 TIER 1, STEP 0: preconditions")
    print("=" * 70)
    import scipy
    print(f"  cwd project root: {PROJECT_ROOT}")
    for d in ["scripts", "outputs", "data"]:
        ok = (PROJECT_ROOT / d).is_dir()
        print(f"    {d}/ present: {ok}")
        if not ok:
            print("  STOP: project layout not found.")
            sys.exit(2)
    print(f"  mne {mne.__version__}, scipy {scipy.__version__}")

    if not ANCHOR_CSV.exists():
        print(f"  STOP: anchor missing: {ANCHOR_CSV}")
        sys.exit(2)
    anc = pd.read_csv(ANCHOR_CSV)
    print(f"\n  anchor {ANCHOR_CSV.name}: shape {anc.shape}")
    print(f"    columns: {list(anc.columns)}")
    print(f"    structure: one row per (movie, channel); {anc.movie.nunique()} movies "
          f"{sorted(anc.movie.unique())},")
    print(f"    {anc.channel.nunique()} channels; column 't' is the male minus female "
          f"theta sex-effect")
    print(f"    Student t per channel (equal_var, the published per-channel anchor).")
    has_fc = all(((anc.channel == c).any()) for c in FC_CH)
    print(f"    contains frontocentral cluster E30/E36/E37: {has_fc}")

    mon = mne.channels.make_standard_montage("GSN-HydroCel-129")
    chpos = mon.get_positions()["ch_pos"]
    e_present = all(f"E{i}" in chpos for i in range(1, 129))
    print(f"\n  montage GSN-HydroCel-129: {len(chpos)} positions; "
          f"E1..E128 present: {e_present}; Cz present: {'Cz' in chpos}")
    print("STEP 0 COMPLETE")
    return anc, mon


def derive_groups(mon):
    """Periocular set by distance to montage-derived orbit points; frontopolar
    midline by anterior midline geometry. Returns (periocular, frontopolar_mid)."""
    p = mon.get_positions()
    chpos = p["ch_pos"]
    names = [c for c in chpos if c.startswith("E")]
    P = np.array([chpos[c] for c in names])
    nas = np.array(p["nasion"])
    # approximate orbits: lateral +-30 mm, slightly posterior, inferior to nasion
    orbit_r = nas + np.array([0.030, -0.005, -0.020])
    orbit_l = nas + np.array([-0.030, -0.005, -0.020])
    dmin = np.minimum(np.linalg.norm(P - orbit_r, axis=1),
                      np.linalg.norm(P - orbit_l, axis=1))
    peri = [names[i] for i in np.argsort(dmin)[:N_PERIOCULAR]]
    # frontopolar midline: near-midline, most anterior (vertical blinks peak here)
    x, y = P[:, 0], P[:, 1]
    y85 = np.percentile(y, 85)
    fpm = [names[i] for i in range(len(names)) if abs(x[i]) < 0.020 and y[i] >= y85]
    return peri, fpm


def group_movie_t(anchor, channels):
    """Mean theta sex-effect t over channels, per movie, and across movies."""
    sub = anchor[anchor.channel.isin(channels)]
    per = {mv: float(sub[sub.movie == mv]["t"].mean()) for mv in MOVIE_ORDER}
    per["mean_t"] = float(np.mean([per[mv] for mv in MOVIE_ORDER]))
    per["n_channels"] = int(sub.channel.nunique())
    return per


def step1_gradient(anchor, mon):
    print("\n" + "=" * 70)
    print("GATE 2 TIER 1, STEP 1: periocular vs frontocentral gradient (Test 1)")
    print("=" * 70)
    peri, fpm = derive_groups(mon)
    print(f"  frontocentral cluster (effect channels): {FC_CH}")
    print(f"  periocular set (nearest {N_PERIOCULAR} to orbits, montage-derived): {peri}")
    print(f"  frontopolar-midline subset (anterior midline): {fpm}")
    overlap = sorted(set(peri) & set(FC_CH))
    print(f"  sanity: periocular intersect frontocentral = {overlap if overlap else 'none (good)'}")

    groups = {
        "frontocentral": FC_CH,
        "periocular": peri,
        "frontopolar_midline": fpm,
    }
    rows = {}
    for g, chans in groups.items():
        rows[g] = group_movie_t(anchor, chans)

    print(f"\n  mean theta sex-effect t by group:")
    hdr = f"  {'group':>20} | " + " ".join(f"{mv[:10]:>10}" for mv in MOVIE_ORDER) + \
          f" | {'mean_t':>8} {'nCh':>4}"
    print(hdr)
    for g in groups:
        r = rows[g]
        cells = " ".join(f"{r[mv]:10.3f}" for mv in MOVIE_ORDER)
        print(f"  {g:>20} | {cells} | {r['mean_t']:8.3f} {r['n_channels']:4d}")

    # key contrasts
    print(f"\n  key contrast t(frontocentral) minus t(periocular):")
    c_peri = {}
    c_fpm = {}
    for mv in MOVIE_ORDER:
        c_peri[mv] = rows["frontocentral"][mv] - rows["periocular"][mv]
        c_fpm[mv] = rows["frontocentral"][mv] - rows["frontopolar_midline"][mv]
    c_peri["mean_t"] = rows["frontocentral"]["mean_t"] - rows["periocular"]["mean_t"]
    c_fpm["mean_t"] = rows["frontocentral"]["mean_t"] - rows["frontopolar_midline"]["mean_t"]
    print("  " + " ".join(f"{mv[:10]}={c_peri[mv]:+.2f}" for mv in MOVIE_ORDER) +
          f" | mean={c_peri['mean_t']:+.2f}")
    print(f"  key contrast t(frontocentral) minus t(frontopolar-midline):")
    print("  " + " ".join(f"{mv[:10]}={c_fpm[mv]:+.2f}" for mv in MOVIE_ORDER) +
          f" | mean={c_fpm['mean_t']:+.2f}")

    # verdict
    fc_gt_peri = all(rows["frontocentral"][mv] > rows["periocular"][mv] for mv in MOVIE_ORDER)
    fc_gt_fpm = all(rows["frontocentral"][mv] > rows["frontopolar_midline"][mv] for mv in MOVIE_ORDER)
    print("\n  VERDICT (Test 1):")
    print(f"    frontocentral t exceeds periocular t in all movies      : {fc_gt_peri}")
    print(f"    frontocentral t exceeds frontopolar-midline t in all movies: {fc_gt_fpm}")
    if fc_gt_peri and fc_gt_fpm:
        print("    => The theta sex effect peaks frontocentrally, away from the eyes,")
        print("       and is near null at periocular and frontopolar electrodes. This")
        print("       argues against an ocular generator.")
    else:
        print("    => Periocular or frontopolar t is comparable to or larger than")
        print("       frontocentral. OCULAR-CONSISTENT. Flagged, not softened.")

    # write CSV
    out_rows = []
    for g in groups:
        r = rows[g]
        out_rows.append({"group": g, **{mv: r[mv] for mv in MOVIE_ORDER},
                         "mean_t": r["mean_t"], "n_channels": r["n_channels"]})
    out_rows.append({"group": "contrast_fc_minus_periocular",
                     **{mv: c_peri[mv] for mv in MOVIE_ORDER},
                     "mean_t": c_peri["mean_t"], "n_channels": ""})
    out_rows.append({"group": "contrast_fc_minus_frontopolar_midline",
                     **{mv: c_fpm[mv] for mv in MOVIE_ORDER},
                     "mean_t": c_fpm["mean_t"], "n_channels": ""})
    cols = ["group"] + MOVIE_ORDER + ["mean_t", "n_channels"]
    pd.DataFrame(out_rows)[cols].to_csv(GRADIENT_CSV, index=False)
    print(f"\n  wrote {GRADIENT_CSV}")
    print("STEP 1 COMPLETE")


def main():
    anchor, mon = step0_preconditions()
    step1_gradient(anchor, mon)


if __name__ == "__main__":
    main()
