"""Figure 2: ISC overview (3 panels).

A: bar chart of grand-mean ISC per movie with 95% CI, RS baseline as dashed line.
B: 4 topomaps of per-channel mean ISC + shared colorbar.
C: log-y bar comparison of RS vs movie ISC with magnitude ratios.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np
import pandas as pd
import mne

from style import (apply_paper_style, MOVIE_ORDER, MOVIE_COLORS, MOVIE_LABELS,
                   get_montage_info, save_pdf_png)

apply_paper_style()


def main():
    isc = pd.read_csv("outputs/R12345678_isc_per_subject.csv")
    rs = pd.read_csv("outputs/R1_resting_state_isc.csv")
    info, ch_names = get_montage_info()

    rs_mean = float(rs.isc_grand_mean.mean())

    fig = plt.figure(figsize=(15, 4.8))
    outer = GridSpec(1, 3, width_ratios=[1.0, 2.2, 1.2], wspace=0.35,
                     left=0.06, right=0.98, top=0.88, bottom=0.22)

    # === Panel A: bar chart with 95% CI ===
    axA = fig.add_subplot(outer[0, 0])
    means, cis, colors = [], [], []
    for m in MOVIE_ORDER:
        sub = isc[isc.movie == m].isc_grand_mean.dropna()
        means.append(sub.mean())
        cis.append(1.96 * sub.std(ddof=1) / np.sqrt(len(sub)))
        colors.append(MOVIE_COLORS[m])
    x = np.arange(len(MOVIE_ORDER))
    axA.bar(x, means, yerr=cis, color=colors, edgecolor="black", linewidth=0.5,
            error_kw={"linewidth": 0.8, "capsize": 3})
    axA.set_xticks(x)
    axA.set_xticklabels([MOVIE_LABELS[m] for m in MOVIE_ORDER],
                        rotation=30, ha="right")
    axA.set_ylabel("Mean ISC")
    axA.set_title("A. Grand-mean ISC by movie (n=1143)", loc="left")
    axA.axhline(rs_mean, color="gray", linestyle="--", linewidth=0.8)
    axA.text(0.99, rs_mean + 0.0015, "RS baseline",
             fontsize=8, color="gray", ha="right",
             transform=axA.get_yaxis_transform())
    axA.set_ylim(0, max(means) * 1.25)

    # === Panel B: 4 topomaps + shared colorbar ===
    inner_b = GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[0, 1],
                                       width_ratios=[1, 1, 1, 1, 0.08],
                                       wspace=0.1)
    ch_cols = [f"isc_{c}" for c in ch_names]
    panel_b_axes = [fig.add_subplot(inner_b[0, i]) for i in range(4)]
    cbar_ax = fig.add_subplot(inner_b[0, 4])
    vmin, vmax = 0.0, 0.15
    last_im = None
    for ax, m in zip(panel_b_axes, MOVIE_ORDER):
        sub = isc[isc.movie == m]
        chan_mean = sub[ch_cols].apply(pd.to_numeric, errors="coerce").mean(axis=0)
        ordered = pd.Series(chan_mean.values, index=ch_names).reindex(info["ch_names"]).values
        im, _ = mne.viz.plot_topomap(ordered, info, axes=ax, show=False,
                                     cmap="Reds", vlim=(vmin, vmax), sensors=False)
        ax.set_title(MOVIE_LABELS[m], fontsize=9, y=-0.18)
        last_im = im
    cbar = fig.colorbar(last_im, cax=cbar_ax, orientation="vertical")
    cbar.set_label("Mean ISC", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    panel_b_axes[0].text(-0.05, 1.18, "B. Per-channel ISC topography",
                         transform=panel_b_axes[0].transAxes, fontsize=12)

    # === Panel C: log-y RS vs movies (bar chart) ===
    axC = fig.add_subplot(outer[0, 2])
    bar_means = [rs_mean] + [isc[isc.movie == m].isc_grand_mean.mean() for m in MOVIE_ORDER]
    bar_labels = ["RS"] + [MOVIE_LABELS[m] for m in MOVIE_ORDER]
    bar_colors = ["#888888"] + [MOVIE_COLORS[m] for m in MOVIE_ORDER]
    xc = np.arange(len(bar_means))
    axC.bar(xc, bar_means, color=bar_colors, edgecolor="black", linewidth=0.5,
            width=0.7)
    axC.set_yscale("log")
    axC.set_xticks(xc)
    axC.set_xticklabels(bar_labels, rotation=45, ha="right", fontsize=9)
    axC.set_ylabel("Mean ISC (log scale)")
    axC.set_title("C. Movie vs resting state", loc="left")
    for i, m in enumerate(MOVIE_ORDER, start=1):
        ratio = bar_means[i] / max(rs_mean, 1e-6)
        axC.text(i, bar_means[i] * 1.18, f"{ratio:.0f}x",
                 ha="center", fontsize=9, fontweight="bold")
    axC.set_ylim(rs_mean * 0.5, max(bar_means) * 3)

    save_pdf_png(fig, "figure2_isc_overview")
    plt.close(fig)
    print("Saved figures/paper/figure2_isc_overview.{pdf,png}")


if __name__ == "__main__":
    main()
