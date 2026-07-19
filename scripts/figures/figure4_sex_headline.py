"""Figure 4: Sex effect headline (5 stacked rows).

A: M vs F bar chart per movie with t/p annotations.
B: 4 broadband sex-effect t-stat topomaps (n=1143).
C: Frequency-band decomposition (theta/alpha/beta sex-effect t per movie).
D: 4 theta-band sex-effect t-stat topomaps (n=400).
E: Sex × age interaction (per-bin M-F diff per movie, with 11-14 highlight).
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

from style import (apply_paper_style, MOVIE_ORDER, MOVIE_COLORS, MOVIE_LABELS,
                   SEX_COLORS, BAND_COLORS, get_montage_info, save_pdf_png)

apply_paper_style()


def _draw_cluster_markers(ax, fc_cluster):
    """Highlight frontocentral cluster channels on a topomap with prominent markers."""
    chmap = mne.channels.make_standard_montage("GSN-HydroCel-129").get_positions()["ch_pos"]
    xs, ys = [], []
    for ch in fc_cluster:
        if ch in chmap:
            xs.append(chmap[ch][0])
            ys.append(chmap[ch][1])
    # Two-layer marker: white outer ring + black inner edge so it pops on any color
    ax.scatter(xs, ys, s=130, marker="o", facecolors="none",
               edgecolors="white", linewidths=2.6, zorder=10)
    ax.scatter(xs, ys, s=130, marker="o", facecolors="none",
               edgecolors="black", linewidths=1.4, zorder=11)


def main():
    isc = pd.read_csv("outputs/R12345678_isc_per_subject.csv")
    master = pd.read_csv("outputs/R12345678_master.csv")
    df_full = isc.merge(
        master[["participant_id", "sex", "age"]].drop_duplicates("participant_id"),
        on="participant_id", how="inner",
    )
    broad = pd.read_csv("outputs/R12345678_sex_effect_per_channel.csv")
    theta = pd.read_csv("outputs/R12345678_sex_effect_theta_per_channel.csv")
    bands = {
        "delta": pd.read_csv("outputs/R12345678_isc_by_band_delta.csv"),
        "theta": pd.read_csv("outputs/R12345678_isc_by_band_theta.csv"),
        "alpha": pd.read_csv("outputs/R12345678_isc_by_band_alpha.csv"),
        "beta":  pd.read_csv("outputs/R12345678_isc_by_band_beta.csv"),
    }
    info, ch_names = get_montage_info()
    fc_cluster = ["E30", "E36", "E37"]

    fig = plt.figure(figsize=(14, 16))
    gs = fig.add_gridspec(5, 4, height_ratios=[1, 1, 1, 1, 1],
                           left=0.06, right=0.93, top=0.97, bottom=0.04,
                           hspace=0.65, wspace=0.3)

    # === Panel A: M vs F bar chart per movie ===
    axA = fig.add_subplot(gs[0, :])
    width = 0.35
    x = np.arange(len(MOVIE_ORDER))
    m_vals, f_vals, m_cis, f_cis, ts, ps = [], [], [], [], [], []
    for movie in MOVIE_ORDER:
        sub = df_full[df_full.movie == movie]
        m = sub[sub.sex == "M"].isc_grand_mean.dropna()
        f = sub[sub.sex == "F"].isc_grand_mean.dropna()
        m_vals.append(m.mean()); f_vals.append(f.mean())
        m_cis.append(1.96 * m.std() / np.sqrt(len(m)))
        f_cis.append(1.96 * f.std() / np.sqrt(len(f)))
        t, p = ttest_ind(m, f); ts.append(t); ps.append(p)
    axA.bar(x - width/2, m_vals, width, yerr=m_cis, color=SEX_COLORS["M"],
            edgecolor="black", linewidth=0.5, label="Male", capsize=3,
            error_kw={"linewidth": 0.8})
    axA.bar(x + width/2, f_vals, width, yerr=f_cis, color=SEX_COLORS["F"],
            edgecolor="black", linewidth=0.5, label="Female", capsize=3,
            error_kw={"linewidth": 0.8})
    for i, (tv, pv) in enumerate(zip(ts, ps)):
        ymax = max(m_vals[i] + m_cis[i], f_vals[i] + f_cis[i])
        exp = int(np.floor(np.log10(pv))) if pv > 0 else -300
        axA.text(i, ymax + 0.005, f"t = {tv:.1f}\np < 10$^{{{exp}}}$",
                 ha="center", fontsize=8.5)
    axA.set_xticks(x)
    axA.set_xticklabels([MOVIE_LABELS[m] for m in MOVIE_ORDER])
    axA.set_ylabel("Mean ISC")
    axA.set_title("A. Grand-mean ISC by sex and movie (n=1143)", loc="left", fontsize=13)
    axA.legend(loc="upper right")
    axA.set_ylim(0, max(m_vals) * 1.45)

    # === Panel B: 4 broadband t-stat topomaps ===
    panel_b_axes = [fig.add_subplot(gs[1, i]) for i in range(4)]
    vmax_b = 20.0
    last_im_b = None
    for ax, m in zip(panel_b_axes, MOVIE_ORDER):
        sub = broad[broad.movie == m].set_index("channel")
        vals = sub["t"].reindex(info["ch_names"]).values
        im, _ = mne.viz.plot_topomap(vals, info, axes=ax, show=False,
                                     cmap="RdBu_r", vlim=(-vmax_b, vmax_b),
                                     sensors=False)
        _draw_cluster_markers(ax, fc_cluster)
        max_t = float(np.nanmax(vals))
        ax.set_title(f"{MOVIE_LABELS[m]}\nmax t = {max_t:.1f}", fontsize=9)
        last_im_b = im
    cbar_ax_b = fig.add_axes([0.94, 0.62, 0.012, 0.12])
    cbar = fig.colorbar(last_im_b, cax=cbar_ax_b, orientation="vertical")
    cbar.set_label("Sex effect t", fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    panel_b_axes[0].text(-0.05, 1.30,
                         "B. Broadband sex-effect topography (n=1143). "
                         "Black/white circles: frontocentral cluster (E30, E36, E37)",
                         transform=panel_b_axes[0].transAxes, fontsize=12)

    # === Panel C: band decomposition ===
    axC = fig.add_subplot(gs[2, :])
    band_t = {b: [] for b in bands}
    for b, df_b in bands.items():
        merged = df_b.merge(
            master[["participant_id", "sex"]].drop_duplicates("participant_id"),
            on="participant_id", how="inner",
        )
        for movie in MOVIE_ORDER:
            sub = merged[merged.movie == movie]
            mm = sub[sub.sex == "M"].isc_grand_mean.dropna()
            ff = sub[sub.sex == "F"].isc_grand_mean.dropna()
            t, _ = ttest_ind(mm, ff)
            band_t[b].append(t)
    bw = 0.2
    # delta placed first so panel C reads in ascending frequency: delta, theta, alpha, beta
    band_list = ["delta", "theta", "alpha", "beta"]
    band_x_offset = {"delta": -1.5 * bw, "theta": -0.5 * bw,
                     "alpha": 0.5 * bw, "beta": 1.5 * bw}
    band_range = {"delta": "1-4", "theta": "4-8", "alpha": "8-12", "beta": "12-20"}
    # delta extends the existing blues gradient one step below theta, so darker
    # still reads as lower frequency. style.py BAND_COLORS is left untouched.
    band_colors = {**BAND_COLORS, "delta": "#08306b"}
    for b in band_list:
        axC.bar(np.arange(len(MOVIE_ORDER)) + band_x_offset[b], band_t[b], bw,
                color=band_colors[b], edgecolor="black", linewidth=0.5,
                label=f"{b} ({band_range[b]} Hz)")
    axC.axhline(2.0, color="gray", linestyle="--", linewidth=0.6)
    axC.axhline(-2.0, color="gray", linestyle="--", linewidth=0.6)
    axC.text(3.55, 2.3, "p ≈ 0.05", fontsize=8, color="gray")
    axC.set_xticks(np.arange(len(MOVIE_ORDER)))
    axC.set_xticklabels([MOVIE_LABELS[m] for m in MOVIE_ORDER])
    axC.set_ylabel("Sex effect t-statistic (M vs F)")
    axC.set_title("C. Sex effect by frequency band (n=400 balanced)",
                  loc="left", fontsize=13)
    axC.legend(loc="upper right", title="Band")

    # === Panel D: 4 theta-band t-stat topomaps ===
    panel_d_axes = [fig.add_subplot(gs[3, i]) for i in range(4)]
    vmax_d = 7.0
    last_im_d = None
    for ax, m in zip(panel_d_axes, MOVIE_ORDER):
        sub = theta[theta.movie == m].set_index("channel")
        vals = sub["t"].reindex(info["ch_names"]).values
        im, _ = mne.viz.plot_topomap(vals, info, axes=ax, show=False,
                                     cmap="RdBu_r", vlim=(-vmax_d, vmax_d),
                                     sensors=False)
        _draw_cluster_markers(ax, fc_cluster)
        max_t = float(np.nanmax(vals))
        ax.set_title(f"{MOVIE_LABELS[m]}\nmax t = {max_t:.1f}", fontsize=9)
        last_im_d = im
    cbar_ax_d = fig.add_axes([0.94, 0.225, 0.012, 0.12])
    cbar = fig.colorbar(last_im_d, cax=cbar_ax_d, orientation="vertical")
    cbar.set_label("Sex effect t", fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    panel_d_axes[0].text(-0.05, 1.30,
                         "D. Theta-band sex-effect topography (n=400 balanced)",
                         transform=panel_d_axes[0].transAxes, fontsize=12)

    # === Panel E: sex × age interaction ===
    axE = fig.add_subplot(gs[4, :])
    age_bins = [(5, 8), (8, 11), (11, 14), (14, 17), (17, 22)]
    bin_labels = ["5-8", "8-11", "11-14", "14-17", "17-22"]
    bin_centers = np.arange(len(bin_labels))
    # Highlight 11-14 bin background. axvspan uses data coords on x; that's fine
    # (bin_centers are 0..4). The earlier bug used data coords on Y before plotting.
    axE.axvspan(1.5, 2.5, color="#fff7c2", alpha=0.6, zorder=0)
    for movie in MOVIE_ORDER:
        diffs, errs = [], []
        for low, high in age_bins:
            sub = df_full[(df_full.movie == movie) & (df_full.age >= low) & (df_full.age < high)]
            mm = sub[sub.sex == "M"].isc_grand_mean.dropna()
            ff = sub[sub.sex == "F"].isc_grand_mean.dropna()
            if len(mm) >= 5 and len(ff) >= 5:
                d = mm.mean() - ff.mean()
                se = np.sqrt(mm.var(ddof=1)/len(mm) + ff.var(ddof=1)/len(ff))
                diffs.append(d); errs.append(1.96 * se)
            else:
                diffs.append(np.nan); errs.append(np.nan)
        axE.errorbar(bin_centers, diffs, yerr=errs, marker="o", markersize=6,
                     color=MOVIE_COLORS[movie], linewidth=1.6, capsize=3,
                     label=MOVIE_LABELS[movie])
    axE.axhline(0, color="black", linewidth=0.4)
    axE.set_xticks(bin_centers)
    axE.set_xticklabels(bin_labels)
    axE.set_xlabel("Age bin (years)")
    axE.set_ylabel("ISC difference (M - F)")
    axE.set_title(
        "E. Sex × age interaction (n=1143). Developmental peak at 11-14 (yellow); "
        "quadratic sex × age² significant in 4/4 movies (all negative)",
        loc="left", fontsize=12,
    )
    # Place "developmental peak" annotation INSIDE Panel E in axes coordinates,
    # not in stale data coords (the original bug).
    axE.text(0.5, 0.96, "developmental peak", transform=axE.transAxes,
             ha="center", va="top", fontsize=9, color="#a37200",
             fontweight="bold")
    axE.legend(loc="upper right", ncol=2)

    save_pdf_png(fig, "figure4_sex_headline")
    fig.savefig("figures/paper/figure4_sex_headline_hires.png", dpi=600)
    plt.close(fig)
    print("Saved figures/paper/figure4_sex_headline.{pdf,png} + _hires.png")


if __name__ == "__main__":
    main()
