"""Figure 3: Developmental effect.

Panel A: 4 scatter plots (one per movie), age vs ISC, sex-colored, with regression line.
Panel B: forest plot of age regression coefficient (with 95% CI) per movie.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy import stats

from style import (apply_paper_style, MOVIE_ORDER, MOVIE_LABELS, MOVIE_COLORS,
                   SEX_COLORS, save_pdf_png)

apply_paper_style()


def main():
    df = pd.read_csv("outputs/R12345678_sex_analysis_df.csv")
    df_full = df.copy()
    df = df.dropna(subset=["ehq_total", "p_factor", "attention", "internalizing", "externalizing"]).copy()
    df["sex_M"] = (df["sex"] == "M").astype(int)
    df["age_sq"] = df["age"] ** 2

    # Compute age coefficient per movie (full controls)
    rows = []
    for m in MOVIE_ORDER:
        sub = df[df.movie == m]
        formula = ("isc_grand_mean ~ sex_M + age + age_sq + ehq_total + C(release) "
                   "+ p_factor + attention + internalizing + externalizing")
        fit = smf.ols(formula, data=sub).fit()
        ci = fit.conf_int().loc["age"]
        rows.append({"movie": m, "beta": fit.params["age"], "p": fit.pvalues["age"],
                     "ci_low": float(ci[0]), "ci_high": float(ci[1]), "n": len(sub)})
    age_results = pd.DataFrame(rows)
    _, age_results["q"], _, _ = multipletests(age_results["p"], method="fdr_bh")

    fig = plt.figure(figsize=(14, 4.4))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 1.0], wspace=0.4)

    # Shared y-axis for the 4 scatter subplots
    y_global_min = float(df_full.isc_grand_mean.min())
    y_global_max = float(df_full.isc_grand_mean.max())
    y_pad = 0.02

    # Panel A: 4 scatter plots
    for i, m in enumerate(MOVIE_ORDER):
        ax = fig.add_subplot(gs[0, i])
        sub = df_full[df_full.movie == m]
        for sx, color in SEX_COLORS.items():
            ss = sub[sub.sex == sx]
            ax.scatter(ss.age, ss.isc_grand_mean, s=8, alpha=0.35, color=color,
                       label=("Male" if sx == "M" else "Female"), edgecolors="none")
        # Regression line + 95% CI band (overall, not per sex)
        x = sub.age.values
        y = sub.isc_grand_mean.values
        slope, intercept, _, _, _ = stats.linregress(x, y)
        xs = np.linspace(x.min(), x.max(), 100)
        ys = slope * xs + intercept
        # Bootstrap CI for the regression line
        rng = np.random.default_rng(0)
        n_boot = 500
        boot_lines = np.zeros((n_boot, len(xs)))
        for b in range(n_boot):
            idx = rng.integers(0, len(x), len(x))
            sl, it, _, _, _ = stats.linregress(x[idx], y[idx])
            boot_lines[b] = sl * xs + it
        lo = np.percentile(boot_lines, 2.5, axis=0)
        hi = np.percentile(boot_lines, 97.5, axis=0)
        ax.fill_between(xs, lo, hi, color="black", alpha=0.15, linewidth=0)
        ax.plot(xs, ys, color="black", linewidth=1.2)

        ax.set_xlim(4.5, 22)
        ax.set_ylim(y_global_min - y_pad, y_global_max + y_pad)
        ax.set_xlabel("Age (years)" if i == 0 else "Age")
        if i == 0:
            ax.set_ylabel("Grand-mean ISC")
        else:
            ax.set_yticklabels([])
        # q label
        q = age_results[age_results.movie == m]["q"].values[0]
        ax.text(0.04, 0.95, f"q$_{{FDR}}$ = {q:.1e}", transform=ax.transAxes,
                fontsize=8, va="top")
        ax.set_title(MOVIE_LABELS[m], fontsize=10)
        if i == 0:
            ax.legend(loc="lower right", fontsize=7, markerscale=1.5)

    # Panel B: forest plot
    axB = fig.add_subplot(gs[0, 4])
    y_pos = np.arange(len(MOVIE_ORDER))[::-1]
    # Extend xlim to leave room for q labels at the right edge
    x_min = float(min(age_results.ci_low)) - 0.001
    x_max = float(max(age_results.ci_high)) + 0.012
    label_x = x_max - 0.001
    for k, (i, row) in enumerate(age_results.iterrows()):
        c = MOVIE_COLORS[row["movie"]]
        axB.errorbar(row["beta"], y_pos[k],
                     xerr=[[row["beta"] - row["ci_low"]], [row["ci_high"] - row["beta"]]],
                     fmt="o", color=c, markersize=7, capsize=4, elinewidth=1.2)
        # Place q label at fixed right-edge position (consistent column),
        # not next to each error bar (which would overlap longer ones).
        axB.text(label_x, y_pos[k], f"q={row['q']:.1e}",
                 va="center", ha="right", fontsize=8)
    axB.axvline(0, color="black", linestyle="--", linewidth=0.6)
    axB.set_yticks(y_pos)
    axB.set_yticklabels([MOVIE_LABELS[m] for m in age_results.movie])
    axB.set_xlabel(r"Age $\beta$ (per year, with controls)")
    axB.set_title("B. Age effect (forest plot)", loc="left")
    axB.set_xlim(x_min, x_max)

    # Panel A label
    fig.text(0.02, 0.95, "A. ISC vs age (per movie)", fontsize=12)

    save_pdf_png(fig, "figure3_developmental")
    plt.close(fig)
    print("Saved figures/paper/figure3_developmental.{pdf,png}")
    print("\nAge coefficients with controls:")
    print(age_results.round({"beta": 5, "p": 4, "q": 4, "ci_low": 5, "ci_high": 5}).to_string(index=False))


if __name__ == "__main__":
    main()
