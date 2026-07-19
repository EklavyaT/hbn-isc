"""Supplementary figures S1-S5."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import ttest_ind

from style import (apply_paper_style, MOVIE_ORDER, MOVIE_LABELS, MOVIE_COLORS,
                   SEX_COLORS, save_pdf_png)

apply_paper_style()


def figure_s1_qc():
    """S1: QC pass rates by release."""
    counts_full = {"R1": 120, "R2": 120, "R3": 157, "R4": 293,
                   "R5": 282, "R6": 102, "R7": 247, "R8": 214}
    counts_pass = {"R1": 117, "R2": 114, "R3": 138, "R4": 259,
                   "R5": 207, "R6": 48, "R7": 124, "R8": 136}
    rels = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"]
    rates = [counts_pass[r] / counts_full[r] * 100 for r in rels]
    colors = ["steelblue"] * 8
    colors[5] = "salmon"  # R6
    colors[6] = "salmon"  # R7

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(rels, rates, color=colors, edgecolor="black", linewidth=0.5)
    for i, (r, rate) in enumerate(zip(rels, rates)):
        ax.text(i, rate + 1.5, f"{counts_pass[r]}/{counts_full[r]}",
                ha="center", fontsize=8)
    ax.axhline(80, color="gray", linestyle="--", linewidth=0.6)
    ax.text(7.7, 82, "80%", fontsize=8, color="gray")
    ax.set_ylabel("QC pass rate (%)")
    ax.set_xlabel("Release")
    ax.set_ylim(0, 105)
    ax.set_title("Figure S1. QC pass rates by release\n"
                 "R6 and R7 (salmon) drop below 65% (R1-R4 stay above 87%)",
                 loc="left")
    save_pdf_png(fig, "figure_S1_qc_by_release")
    plt.close(fig)
    print("Saved S1: QC pass rates")


def figure_s2_robustness():
    """S2: Sex effect under age / CBCL / joint matching (3 panels)."""
    df = pd.read_csv("outputs/R12345678_sex_analysis_df.csv").dropna(subset=["attention"])

    rng = np.random.default_rng(42)

    def matched_t(movie, age_bin=False, attn_bin=False):
        sub = df[df.movie == movie].copy()
        if age_bin:
            sub["age_bin"] = pd.cut(sub.age, bins=[5, 8, 11, 14, 17, 22], labels=False)
        if attn_bin:
            sub["attn_bin"] = pd.qcut(sub.attention, q=5, labels=False, duplicates="drop")

        groups = []
        if age_bin and attn_bin:
            for ab in sub.age_bin.dropna().unique():
                for tb in sub.attn_bin.dropna().unique():
                    groups.append(sub[(sub.age_bin == ab) & (sub.attn_bin == tb)])
        elif age_bin:
            for ab in sub.age_bin.dropna().unique():
                groups.append(sub[sub.age_bin == ab])
        elif attn_bin:
            for tb in sub.attn_bin.dropna().unique():
                groups.append(sub[sub.attn_bin == tb])
        else:
            groups = [sub]

        matched = []
        for g in groups:
            ms, fs = g[g.sex == "M"], g[g.sex == "F"]
            n = min(len(ms), len(fs))
            if n > 0:
                idx_m = rng.choice(ms.index, n, replace=False)
                idx_f = rng.choice(fs.index, n, replace=False)
                matched.append(df.loc[idx_m]); matched.append(df.loc[idx_f])
        if not matched:
            return np.nan, 0
        mdf = pd.concat(matched)
        m = mdf[mdf.sex == "M"].isc_grand_mean.dropna()
        f = mdf[mdf.sex == "F"].isc_grand_mean.dropna()
        if len(m) < 5 or len(f) < 5:
            return np.nan, 0
        t, _ = ttest_ind(m, f)
        return t, len(mdf)

    rows = []
    for label, age_b, attn_b in [("Unmatched", False, False),
                                  ("Age-matched", True, False),
                                  ("CBCL-matched", False, True),
                                  ("Joint-matched", True, True)]:
        for movie in MOVIE_ORDER:
            t, n = matched_t(movie, age_b, attn_b)
            rows.append({"condition": label, "movie": movie, "t": t, "n": n})
    res = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, cond in zip(axes, ["Age-matched", "CBCL-matched", "Joint-matched"]):
        sub = res[res.condition == cond]
        unm = res[res.condition == "Unmatched"]
        x = np.arange(len(MOVIE_ORDER))
        ax.bar(x, sub["t"].values,
               color=[MOVIE_COLORS[m] for m in MOVIE_ORDER],
               edgecolor="black", linewidth=0.5)
        for i, m in enumerate(MOVIE_ORDER):
            ax.scatter(i, unm[unm.movie == m].iloc[0]["t"], marker="_",
                       color="black", s=200, zorder=10, linewidths=2)
        ax.set_xticks(x)
        ax.set_xticklabels([MOVIE_LABELS[m] for m in MOVIE_ORDER],
                           rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{cond} (n={int(sub.n.mean())} per movie)")
        ax.axhline(0, color="black", linewidth=0.4)
        ax.axhline(2, color="gray", linestyle="--", linewidth=0.6)
    axes[0].set_ylabel("Sex effect t-statistic")
    fig.suptitle("Figure S2. Sex effect under matching schemes\n"
                 "Black tick = unmatched (full cohort) reference; bars = matched cohort", y=1.02)
    save_pdf_png(fig, "figure_S2_robustness")
    plt.close(fig)
    print("Saved S2: robustness matching")


def figure_s3_qc_sensitivity():
    """S3: Sex effect under tightening QC."""
    df = pd.read_csv("outputs/R12345678_sex_analysis_df.csv")

    def t_for_movie(d, movie):
        sub = d[d.movie == movie]
        m = sub[sub.sex == "M"].isc_grand_mean.dropna()
        f = sub[sub.sex == "F"].isc_grand_mean.dropna()
        if len(m) < 10 or len(f) < 10:
            return np.nan, 0
        t, _ = ttest_ind(m, f)
        return t, sub.participant_id.nunique()

    qc_levels = [
        ("loose (full)", df, df.participant_id.nunique()),
        ("strict",
         df[(df.max_std_p50_uV <= 20) & (df.min_icalabel_median_conf >= 0.65)
            & (df.min_ic_brain >= 6)],
         None),
        ("very strict",
         df[(df.max_std_p50_uV <= 15) & (df.min_icalabel_median_conf >= 0.70)
            & (df.min_ic_brain >= 8)],
         None),
    ]

    rows = []
    for label, sub_df, _ in qc_levels:
        for movie in MOVIE_ORDER:
            t, n = t_for_movie(sub_df, movie)
            rows.append({"qc": label, "movie": movie, "t": t,
                         "n": sub_df.participant_id.nunique()})
    res = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    width = 0.27
    for k, (label, _, _) in enumerate(qc_levels):
        sub = res[res.qc == label]
        x = np.arange(len(MOVIE_ORDER)) + (k - 1) * width
        ax.bar(x, sub["t"].values, width, label=f"{label} (n={int(sub.n.iloc[0])})",
               edgecolor="black", linewidth=0.5)
    ax.axhline(2, color="gray", linestyle="--", linewidth=0.6)
    ax.text(3.6, 2.2, "p ≈ 0.05", fontsize=8, color="gray")
    ax.set_xticks(np.arange(len(MOVIE_ORDER)))
    ax.set_xticklabels([MOVIE_LABELS[m] for m in MOVIE_ORDER])
    ax.set_ylabel("Sex effect t-statistic")
    ax.set_title("Figure S3. Sex effect under tightening QC\n"
                 "Effect attenuates ~40% at strict QC; loses significance for 3/4 movies at very strict (small n)",
                 loc="left")
    ax.legend(title="QC level")
    save_pdf_png(fig, "figure_S3_qc_sensitivity")
    plt.close(fig)
    print("Saved S3: QC sensitivity")


def figure_s4_sex_movie_interaction():
    """S4: Sex × movie interaction forest plot of marginal sex coefficients."""
    df = pd.read_csv("outputs/R12345678_sex_analysis_df.csv").dropna(subset=["ehq_total"])
    df["sex_M"] = (df["sex"] == "M").astype(int)
    df["age_sq"] = df["age"] ** 2
    rows = []
    for movie in MOVIE_ORDER:
        sub = df[df.movie == movie]
        formula = ("isc_grand_mean ~ sex_M + age + age_sq + ehq_total + C(release) "
                   "+ p_factor + attention + internalizing + externalizing")
        fit = smf.ols(formula, data=sub).fit()
        ci = fit.conf_int().loc["sex_M"]
        rows.append({"movie": movie, "beta": float(fit.params["sex_M"]),
                     "ci_low": float(ci[0]), "ci_high": float(ci[1]),
                     "p": float(fit.pvalues["sex_M"])})
    res = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 4))
    y = np.arange(len(MOVIE_ORDER))[::-1]
    for k, (i, r) in enumerate(res.iterrows()):
        ax.errorbar(r["beta"], y[k],
                    xerr=[[r["beta"] - r["ci_low"]], [r["ci_high"] - r["beta"]]],
                    fmt="o", color=MOVIE_COLORS[r["movie"]], markersize=10,
                    capsize=5, elinewidth=1.5)
        ax.text(r["ci_high"] + 0.001, y[k],
                f"β = {r['beta']:.4f}, p = {r['p']:.1e}",
                va="center", fontsize=8)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([MOVIE_LABELS[m] for m in res.movie])
    ax.set_xlabel(r"Sex effect $\beta$ (M coded as 1; with controls)")
    ax.set_title("Figure S4. Sex × movie interaction (marginal sex coefficients per movie)\n"
                 "Joint F(3, 4507) = 22.88, p = 1e-14; magnitude varies but direction is uniform",
                 loc="left")
    ax.set_xlim(0, max(res.ci_high) * 1.5)
    save_pdf_png(fig, "figure_S4_sex_movie_interaction")
    plt.close(fig)
    print("Saved S4: sex × movie interaction")


def figure_s5_release_effect():
    """S5: Release effect ANOVA (boxplots per movie)."""
    df = pd.read_csv("outputs/R12345678_sex_analysis_df.csv")
    rels = sorted(df.release.unique())

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6), sharey=True)
    plt.subplots_adjust(top=0.78)
    for ax, movie in zip(axes, MOVIE_ORDER):
        sub = df[df.movie == movie]
        data = [sub[sub.release == r].isc_grand_mean.values for r in rels]
        bp = ax.boxplot(data, tick_labels=rels, patch_artist=True,
                        medianprops={"color": "black", "linewidth": 1.2},
                        flierprops={"marker": ".", "markersize": 3, "alpha": 0.4})
        for box in bp["boxes"]:
            box.set(facecolor=MOVIE_COLORS[movie], alpha=0.6, edgecolor="black")
        from scipy.stats import f_oneway
        F, p = f_oneway(*data)
        ax.set_title(f"{MOVIE_LABELS[movie]}\nANOVA F={F:.1f}, p={p:.1e}", fontsize=10)
        ax.set_xlabel("Release")
    axes[0].set_ylabel("Grand-mean ISC")
    # Single-line suptitle, placed well above the subplot titles
    fig.suptitle(
        "Figure S5. Release effect: ISC distribution by release per movie. "
        "Significant in all 4 movies at R1-R8 (none at R1-R4).",
        y=0.99, fontsize=12,
    )
    save_pdf_png(fig, "figure_S5_release_effect")
    plt.close(fig)
    print("Saved S5: release effect")


if __name__ == "__main__":
    figure_s1_qc()
    figure_s2_robustness()
    figure_s3_qc_sensitivity()
    figure_s4_sex_movie_interaction()
    figure_s5_release_effect()
