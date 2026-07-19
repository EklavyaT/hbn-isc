"""Figure 5: CBCL null + encoding bound.

Panel A: heatmap of CBCL × movie p-values (with q_FDR annotations).
Panel B: encoding model topomaps (LOWLEVEL r + CLIP advantage diff).
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
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from style import (apply_paper_style, MOVIE_ORDER, MOVIE_LABELS,
                   get_montage_info, save_pdf_png)

apply_paper_style()


def main():
    df = pd.read_csv("outputs/R12345678_sex_analysis_df.csv")
    df["sex_M"] = (df["sex"] == "M").astype(int)
    df["age_sq"] = df["age"] ** 2

    factors = ["p_factor", "attention", "internalizing", "externalizing"]
    factor_labels = ["p-factor", "Attention", "Internalizing", "Externalizing"]

    # CBCL × movie p-values
    rows = []
    for movie in MOVIE_ORDER:
        for factor in factors:
            sub = df[df.movie == movie].dropna(subset=[factor, "ehq_total"])
            formula = (f"isc_grand_mean ~ {factor} + sex_M + age + age_sq "
                       "+ ehq_total + C(release)")
            fit = smf.ols(formula, data=sub).fit()
            rows.append({"movie": movie, "factor": factor,
                         "p": float(fit.pvalues[factor]),
                         "coef": float(fit.params[factor])})
    cbcl = pd.DataFrame(rows)
    _, cbcl["q"], _, _ = multipletests(cbcl["p"], method="fdr_bh")

    # encoding results
    enc = pd.read_csv("outputs/trf_banded_30subj_results.csv")
    pivot_low = enc[enc.model == "LOWLEVEL"].groupby("channel")["r_cv_mean"].mean()
    pivot_comb = enc[enc.model == "COMBINED"].groupby("channel")["r_cv_mean"].mean()
    advantage = (pivot_comb - pivot_low).reindex(pivot_low.index)

    info, ch_names = get_montage_info()

    fig = plt.figure(figsize=(14, 5.5))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.2, 0.05, 0.9, 0.9, 0.06], wspace=0.4)

    # Panel A: heatmap (4 factors x 4 movies). p-value as -log10(p), color.
    axA = fig.add_subplot(gs[0, 0])
    grid = np.zeros((len(factors), len(MOVIE_ORDER)))
    for j, m in enumerate(MOVIE_ORDER):
        for i, f in enumerate(factors):
            r = cbcl[(cbcl.movie == m) & (cbcl.factor == f)].iloc[0]
            grid[i, j] = -np.log10(r["p"])
    im = axA.imshow(grid, cmap="Reds", vmin=0, vmax=2.0, aspect="auto")
    axA.set_xticks(range(len(MOVIE_ORDER)))
    axA.set_xticklabels([MOVIE_LABELS[m] for m in MOVIE_ORDER], rotation=30, ha="right")
    axA.set_yticks(range(len(factors)))
    axA.set_yticklabels(factor_labels)
    axA.set_title("A. CBCL × movie ISC: -log10(p) and FDR q-value\n0/16 cells survive q$_{FDR}$ < 0.05", loc="left")
    for i in range(len(factors)):
        for j in range(len(MOVIE_ORDER)):
            r = cbcl[(cbcl.movie == MOVIE_ORDER[j]) & (cbcl.factor == factors[i])].iloc[0]
            txt = f"p={r['p']:.2f}\nq={r['q']:.2f}"
            color = "white" if grid[i, j] > 1.0 else "black"
            axA.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)
    cbar_ax = fig.add_subplot(gs[0, 1])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("-log10(p)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Panel B: encoding topomaps
    axB1 = fig.add_subplot(gs[0, 2])
    axB2 = fig.add_subplot(gs[0, 3])
    cbar_b_ax = fig.add_subplot(gs[0, 4])

    # LOWLEVEL r
    low_vals = pivot_low.reindex(info["ch_names"]).values
    vmax_l = float(np.nanmax(np.abs(low_vals))) if not np.all(np.isnan(low_vals)) else 0.05
    im_l, _ = mne.viz.plot_topomap(low_vals, info, axes=axB1, show=False,
                                    cmap="RdBu_r", vlim=(-vmax_l, vmax_l), sensors=False)
    axB1.set_title(f"LOWLEVEL encoding r\n(mean across 30 subj)", fontsize=10)

    # CLIP advantage
    adv_vals = advantage.reindex(info["ch_names"]).values
    vmax_a = float(np.nanmax(np.abs(adv_vals))) if not np.all(np.isnan(adv_vals)) else 0.01
    im_a, _ = mne.viz.plot_topomap(adv_vals, info, axes=axB2, show=False,
                                    cmap="RdBu_r", vlim=(-vmax_a, vmax_a), sensors=False)
    axB2.set_title(f"CLIP advantage\n(COMBINED - LOWLEVEL)", fontsize=10)
    axB2.text(0.5, -0.15,
              "0/129 channels q$_{FDR}$ < 0.05 with effect > 0.005",
              transform=axB2.transAxes, ha="center", fontsize=9, color="darkred")

    cbar = fig.colorbar(im_a, cax=cbar_b_ax)
    cbar.set_label("Encoding r", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    fig.text(0.55, 0.95, "B. Encoding-model bounds (n=30 subj, ThePresent)",
             fontsize=12)

    save_pdf_png(fig, "figure5_cbcl_encoding_bounds")
    plt.close(fig)
    print("Saved figures/paper/figure5_cbcl_encoding_bounds.{pdf,png}")
    print("\nCBCL p-values:")
    print(cbcl[["movie", "factor", "p", "q"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
