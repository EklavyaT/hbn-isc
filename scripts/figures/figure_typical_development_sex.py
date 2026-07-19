"""Topographic sex effect on the low-symptom CBCL-proxy subset (A1).

Mirrors Figure 4 Panel B (broadband sex-effect topography) but on the n=312
low-symptom subset (M=195, F=117). Caveat: HBN public BIDS releases ship no
DSM diagnostic codes; "low-symptom" here means all 4 CBCL bifactor scores
< 0.5 SD above the HBN sample mean, NOT a population-normed
typical-development definition.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mne
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

from style import (apply_paper_style, MOVIE_ORDER, MOVIE_LABELS,
                   get_montage_info, save_pdf_png)

apply_paper_style()


def main():
    isc = pd.read_csv("outputs/R12345678_isc_per_subject.csv")
    master = pd.read_csv("outputs/R12345678_master.csv")
    df = isc.merge(master[["participant_id", "sex", "p_factor", "attention",
                           "internalizing", "externalizing"]]
                   .drop_duplicates("participant_id"),
                   on="participant_id", how="inner")
    df = df.dropna(subset=["p_factor", "attention", "internalizing", "externalizing"])
    mask = ((df.p_factor < 0.5) & (df.attention < 0.5)
            & (df.internalizing < 0.5) & (df.externalizing < 0.5))
    sub = df[mask].copy()
    print(f"Low-symptom CBCL subset: {sub.participant_id.nunique()} subjects "
          f"(M={sub[sub.sex=='M'].participant_id.nunique()}, "
          f"F={sub[sub.sex=='F'].participant_id.nunique()})")

    info, ch_names = get_montage_info()
    fc_cluster = ["E30", "E36", "E37"]
    chmap = mne.channels.make_standard_montage("GSN-HydroCel-129").get_positions()["ch_pos"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    last_im = None
    for ax, movie in zip(axes, MOVIE_ORDER):
        m_sub = sub[sub.movie == movie]
        rows = []
        for ch in ch_names:
            col = f"isc_{ch}"
            if col not in m_sub.columns:
                continue
            mm = pd.to_numeric(m_sub[m_sub.sex == "M"][col], errors="coerce").dropna()
            ff = pd.to_numeric(m_sub[m_sub.sex == "F"][col], errors="coerce").dropna()
            if len(mm) < 20 or len(ff) < 20:
                continue
            t, p = ttest_ind(mm, ff)
            rows.append({"channel": ch, "t": float(t), "p": float(p)})
        rdf = pd.DataFrame(rows)
        _, q, _, _ = multipletests(rdf["p"], method="fdr_bh")
        rdf["q"] = q

        vals = rdf.set_index("channel")["t"].reindex(info["ch_names"]).values
        vmax = float(np.nanmax(np.abs(vals)))
        im, _ = mne.viz.plot_topomap(vals, info, axes=ax, show=False,
                                     cmap="RdBu_r", vlim=(-vmax, vmax),
                                     sensors=False)
        # Frontocentral cluster markers
        xs, ys = [], []
        for ch in fc_cluster:
            if ch in chmap:
                xs.append(chmap[ch][0]); ys.append(chmap[ch][1])
        ax.scatter(xs, ys, s=130, marker="o", facecolors="none",
                   edgecolors="white", linewidths=2.6, zorder=10)
        ax.scatter(xs, ys, s=130, marker="o", facecolors="none",
                   edgecolors="black", linewidths=1.4, zorder=11)
        n_pos = int(((rdf["q"] < 0.05) & (rdf["t"] > 0)).sum())
        ax.set_title(f"{MOVIE_LABELS[movie]}\nmax t={vmax:.1f}; "
                     f"FDR-pos={n_pos}/{len(rdf)}", fontsize=9)
        last_im = im

    cbar_ax = fig.add_axes([0.94, 0.30, 0.012, 0.42])
    cbar = fig.colorbar(last_im, cax=cbar_ax, orientation="vertical")
    cbar.set_label("Sex effect t (M vs F)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle("Sex-effect topography on the low-symptom CBCL-proxy subset "
                 "(n=312; M=195, F=117). White/black circles: frontocentral "
                 "cluster (E30, E36, E37). "
                 "CAVEAT: HBN BIDS releases ship no DSM diagnoses; this is "
                 "a CBCL-bifactor proxy, not a typical-development cohort.",
                 y=1.02, fontsize=10)
    save_pdf_png(fig, "figure_typical_development_sex")
    plt.close(fig)
    print("Saved figures/paper/figure_typical_development_sex.{pdf,png}")


if __name__ == "__main__":
    main()
