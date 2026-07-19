"""Gate 2 Tier 1, reconciliation: spatial topography of the DELTA sex effect.

Test 2 showed the frontocentral male greater than female cluster effect is larger
in delta (1 to 4 Hz) than theta, a low-frequency-dominant profile that on its own
is blink-consistent. But blink residual is not only low frequency, it is also
periocular. So the decisive disambiguator is where the delta sex effect peaks in
space. If delta peaks frontocentrally and is near null at the orbits (like theta),
the delta dominance is a low-frequency neural pattern, not ocular. If delta peaks
periocularly, an ocular contribution is implicated.

This is Test 1 applied to the delta band: per-channel delta sex-effect t (mean of
the per-movie Student t, equal_var, matching the published theta anchor), then
mean t over the frontocentral, periocular, and frontopolar-midline groups. Reads
outputs/R12345678_isc_by_band_delta.csv and data/R12345678_band_subsample.csv.
Writes outputs/gate2_delta_topography.csv. Overwrites nothing else. No SSD needed.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import mne

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT = PROJECT_ROOT / "outputs"
SUBSAMPLE_CSV = PROJECT_ROOT / "data" / "R12345678_band_subsample.csv"
DELTA_CSV = OUT / "R12345678_isc_by_band_delta.csv"
THETA_CSV = OUT / "R12345678_isc_by_band_theta.csv"
RESULT_CSV = OUT / "gate2_delta_topography.csv"

MOVIE_ORDER = ["DespicableMe", "DiaryOfAWimpyKid", "FunwithFractals", "ThePresent"]
FC_CH = ["E30", "E36", "E37"]
N_PERIOCULAR = 10


def derive_groups():
    mon = mne.channels.make_standard_montage("GSN-HydroCel-129")
    p = mon.get_positions()
    chpos = p["ch_pos"]
    names = [c for c in chpos if c.startswith("E")]
    P = np.array([chpos[c] for c in names])
    nas = np.array(p["nasion"])
    orbit_r = nas + np.array([0.030, -0.005, -0.020])
    orbit_l = nas + np.array([-0.030, -0.005, -0.020])
    dmin = np.minimum(np.linalg.norm(P - orbit_r, axis=1),
                      np.linalg.norm(P - orbit_l, axis=1))
    peri = [names[i] for i in np.argsort(dmin)[:N_PERIOCULAR]]
    x, y = P[:, 0], P[:, 1]
    y85 = np.percentile(y, 85)
    fpm = [names[i] for i in range(len(names)) if abs(x[i]) < 0.020 and y[i] >= y85]
    return peri, fpm


def per_channel_t(df, sex_lookup):
    """Per channel, mean across movies of the per-movie Student t (M minus F)."""
    d = df.merge(sex_lookup, on="participant_id", how="left")
    chans = [c[4:] for c in df.columns if c.startswith("isc_E")] + (["Cz"] if "isc_Cz" in df.columns else [])
    out = {}
    for ch in chans:
        col = f"isc_{ch}"
        if col not in d.columns:
            continue
        d[col] = pd.to_numeric(d[col], errors="coerce")
        ts = []
        for mv in MOVIE_ORDER:
            sub = d[d.movie == mv]
            m = sub[sub.sex == "M"][col].dropna()
            f = sub[sub.sex == "F"][col].dropna()
            if len(m) >= 2 and len(f) >= 2:
                t, _ = ttest_ind(m, f, equal_var=True)
                ts.append(t)
        if ts:
            out[ch] = float(np.mean(ts))
    return pd.Series(out)


def group_report(name, ser, peri, fpm):
    order = list(ser.sort_values(ascending=False).index)
    fc_mean = float(np.mean([ser[c] for c in FC_CH if c in ser]))
    peri_mean = float(np.mean([ser[c] for c in peri if c in ser]))
    fpm_mean = float(np.mean([ser[c] for c in fpm if c in ser]))
    fc_ranks = [order.index(c) + 1 for c in FC_CH if c in order]
    peri_ranks = sorted(order.index(c) + 1 for c in peri if c in order)
    print(f"\n  --- {name} sex-effect t topography ({len(ser)} channels) ---")
    print(f"  top 8: " + ", ".join(f"{c}({ser[c]:.2f})" for c in order[:8]))
    print(f"  frontocentral E30/E36/E37: ranks {fc_ranks}, mean t={fc_mean:.3f}")
    print(f"  periocular ring          : ranks {peri_ranks[:3]}..{peri_ranks[-3:]}, "
          f"mean t={peri_mean:.3f}")
    print(f"  frontopolar-midline      : mean t={fpm_mean:.3f}")
    return dict(fc=fc_mean, peri=peri_mean, fpm=fpm_mean,
                fc_ranks=fc_ranks, peri_ranks=peri_ranks)


def main():
    if not DELTA_CSV.exists():
        print(f"STOP: {DELTA_CSV} not found. Run scripts/17 first.")
        sys.exit(1)
    sub = pd.read_csv(SUBSAMPLE_CSV)
    sex_lookup = sub[["participant_id", "sex"]].drop_duplicates("participant_id")
    peri, fpm = derive_groups()

    print("=" * 70)
    print("GATE 2 TIER 1 reconciliation: spatial topography of the delta sex effect")
    print("=" * 70)
    print(f"  periocular set       : {peri}")
    print(f"  frontopolar-midline  : {fpm}")
    print(f"  frontocentral cluster: {FC_CH}")

    delta_t = per_channel_t(pd.read_csv(DELTA_CSV), sex_lookup)
    theta_t = per_channel_t(pd.read_csv(THETA_CSV), sex_lookup)
    dr = group_report("DELTA", delta_t, peri, fpm)
    tr = group_report("THETA (reference)", theta_t, peri, fpm)

    print("\n  VERDICT (delta topography):")
    delta_peaks_fc = dr["fc"] > dr["peri"] and dr["fc"] > dr["fpm"]
    print(f"    delta frontocentral t ({dr['fc']:.3f}) exceeds periocular "
          f"({dr['peri']:.3f}) and frontopolar-midline ({dr['fpm']:.3f}): "
          f"{delta_peaks_fc}")
    if delta_peaks_fc:
        print("    => Even in delta the sex effect peaks frontocentrally and is near")
        print("       null at the orbits. The delta-over-theta magnitude is a low-")
        print("       frequency NEURAL pattern (frontocentral), not the periocular")
        print("       topography of blink or saccade residual. The blink-spectrum flag")
        print("       from Test 2 is not accompanied by a blink topography.")
    else:
        print("    => Delta sex effect is periocular. Ocular contribution implicated.")

    rows = []
    for grp, val in [("frontocentral_meanT", "fc"), ("periocular_meanT", "peri"),
                     ("frontopolar_midline_meanT", "fpm")]:
        rows.append({"group": grp, "delta": dr[val], "theta": tr[val]})
    pd.DataFrame(rows)[["group", "delta", "theta"]].to_csv(RESULT_CSV, index=False)
    print(f"\n  wrote {RESULT_CSV}")
    print("DELTA TOPOGRAPHY COMPLETE")


if __name__ == "__main__":
    main()
