"""Build all paper tables (CSV / LaTeX booktabs / Markdown).

Outputs:
  tables/csv/table{N}_{name}.csv
  tables/latex/table{N}_{name}.tex
  tables/markdown/table{N}_{name}.md
  tables/INDEX.md

Precision conventions are centralized in `format_*` helpers.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import ttest_ind, f_oneway
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TBL_DIR = PROJECT_ROOT / "tables"
CSV_DIR = TBL_DIR / "csv"
TEX_DIR = TBL_DIR / "latex"
MD_DIR = TBL_DIR / "markdown"
for d in [CSV_DIR, TEX_DIR, MD_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MOVIE_ORDER = ["DespicableMe", "DiaryOfAWimpyKid", "FunwithFractals", "ThePresent"]
MOVIE_LABELS = {
    "DespicableMe": "Despicable Me",
    "DiaryOfAWimpyKid": "Diary of a Wimpy Kid",
    "FunwithFractals": "Fun with Fractals",
    "ThePresent": "The Present",
}


# === Number formatting helpers (paper precision conventions) ===

def fmt_isc(x):
    return f"{x:.3f}" if pd.notnull(x) else ""


def fmt_t(x):
    return f"{x:+.1f}" if pd.notnull(x) else ""


def fmt_p(x):
    if pd.isnull(x):
        return ""
    if x < 0.001:
        return f"{x:.1e}"
    return f"{x:.3f}"


def fmt_q(x):
    if pd.isnull(x):
        return ""
    return f"{x:.2e}"


def fmt_beta(x):
    return f"{x:+.4f}" if pd.notnull(x) else ""


def fmt_ci(lo, hi):
    if pd.isnull(lo) or pd.isnull(hi):
        return ""
    return f"[{lo:+.4f}, {hi:+.4f}]"


def fmt_int(x):
    return f"{int(x)}" if pd.notnull(x) else ""


def fmt_pct(x):
    return f"{x:.1f}%" if pd.notnull(x) else ""


def fmt_mean_sd(mean, sd):
    if pd.isnull(mean):
        return ""
    return f"{mean:.2f} ({sd:.2f})"


def fmt_range(lo, hi):
    return f"{lo:.1f}-{hi:.1f}" if pd.notnull(lo) and pd.notnull(hi) else ""


# === Output writers ===

def df_to_md(df: pd.DataFrame) -> str:
    """Manual GitHub-flavored markdown table (no tabulate dep)."""
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [header, sep]
    for _, row in df.iterrows():
        cells = [str(row[c]) if pd.notnull(row[c]) else "" for c in df.columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def df_to_latex_booktabs(df: pd.DataFrame, caption: str, label: str) -> str:
    cols = list(df.columns)
    col_align = "l" + "r" * (len(cols) - 1)
    out = []
    out.append(r"\begin{table*}[t]")
    out.append(r"\centering")
    out.append(r"\caption{" + caption + "}")
    out.append(r"\label{" + label + "}")
    out.append(r"\begin{tabular}{" + col_align + "}")
    out.append(r"\toprule")
    out.append(" & ".join(_latex_escape(c) for c in cols) + r" \\")
    out.append(r"\midrule")
    for _, row in df.iterrows():
        out.append(" & ".join(_latex_escape(str(row[c])) for c in cols) + r" \\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(r"\end{table*}")
    return "\n".join(out) + "\n"


def _latex_escape(s: str) -> str:
    repl = {
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "<": r"$<$",
        ">": r"$>$",
    }
    out = s
    for k, v in repl.items():
        out = out.replace(k, v)
    return out


def write_table(df: pd.DataFrame, num: int, name: str, caption: str, label: str):
    base = f"table{num}_{name}"
    df.to_csv(CSV_DIR / f"{base}.csv", index=False)
    (TEX_DIR / f"{base}.tex").write_text(df_to_latex_booktabs(df, caption, label))
    (MD_DIR / f"{base}.md").write_text(f"## {caption}\n\n{df_to_md(df)}\n")
    print(f"  wrote {base}.{{csv,tex,md}}", flush=True)


# === Common loaders ===

def _load_master_cohort():
    cohort = pd.read_csv("outputs/R12345678_isc_cohort.csv")
    master = pd.read_csv("outputs/R12345678_master.csv")
    return cohort, master


def _load_analysis_df():
    df = pd.read_csv("outputs/R12345678_sex_analysis_df.csv")
    df["sex_M"] = (df["sex"] == "M").astype(int)
    df["age_sq"] = df["age"] ** 2
    return df


# === TABLE 1: cohort demographics ===

def table1_demographics():
    cohort, master = _load_master_cohort()
    rows = []
    for rel in ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"]:
        sub = cohort[cohort.release == rel]
        n = len(sub)
        n_m = (sub.sex == "M").sum()
        n_f = (sub.sex == "F").sum()
        rows.append({
            "Release": rel,
            "n total": fmt_int(n),
            "n male": fmt_int(n_m),
            "n female": fmt_int(n_f),
            "Age mean (SD)": fmt_mean_sd(sub.age.mean(), sub.age.std()),
            "Age range": fmt_range(sub.age.min(), sub.age.max()),
            "p-factor mean (SD)": fmt_mean_sd(sub.p_factor.mean(), sub.p_factor.std()),
            "Attention mean (SD)": fmt_mean_sd(sub.attention.mean(), sub.attention.std()),
            "Internalizing mean (SD)": fmt_mean_sd(sub.internalizing.mean(), sub.internalizing.std()),
            "Externalizing mean (SD)": fmt_mean_sd(sub.externalizing.mean(), sub.externalizing.std()),
        })
    # Total row
    sub = cohort
    rows.append({
        "Release": "Total",
        "n total": fmt_int(len(sub)),
        "n male": fmt_int((sub.sex == "M").sum()),
        "n female": fmt_int((sub.sex == "F").sum()),
        "Age mean (SD)": fmt_mean_sd(sub.age.mean(), sub.age.std()),
        "Age range": fmt_range(sub.age.min(), sub.age.max()),
        "p-factor mean (SD)": fmt_mean_sd(sub.p_factor.mean(), sub.p_factor.std()),
        "Attention mean (SD)": fmt_mean_sd(sub.attention.mean(), sub.attention.std()),
        "Internalizing mean (SD)": fmt_mean_sd(sub.internalizing.mean(), sub.internalizing.std()),
        "Externalizing mean (SD)": fmt_mean_sd(sub.externalizing.mean(), sub.externalizing.std()),
    })
    df = pd.DataFrame(rows)
    cap = ("Table 1. Demographic and clinical characteristics of the analytic cohort by HBN release. "
           "CBCL bifactor scores are standardized scores from the bifactor decomposition of CBCL items.")
    write_table(df, 1, "demographics", cap, "tab:demographics")


# === TABLE 2: ISC magnitudes ===

def _bootstrap_ci_mean(x, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    boots = np.empty(n_boot)
    n = len(x)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = x[idx].mean()
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def table2_isc_magnitudes():
    per_subj = pd.read_csv("outputs/R12345678_isc_per_subject.csv")
    rs = pd.read_csv("outputs/R1_resting_state_isc.csv")
    ch_cols = [c for c in per_subj.columns if c.startswith("isc_E") or c == "isc_Cz"]
    rows = []
    for movie in MOVIE_ORDER:
        sub = per_subj[per_subj.movie == movie]
        x = sub.isc_grand_mean.dropna().values
        ci_lo, ci_hi = _bootstrap_ci_mean(x)
        # Top channel by per-channel mean ISC
        chan_mean = sub[ch_cols].apply(pd.to_numeric, errors="coerce").mean(axis=0)
        top_ch_full = chan_mean.idxmax()
        top_ch = top_ch_full.replace("isc_", "")
        top_val = chan_mean.max()
        rows.append({
            "Stimulus": MOVIE_LABELS[movie],
            "n": fmt_int(len(x)),
            "Mean ISC (95\\% CI)": f"{x.mean():.3f} [{ci_lo:.3f}, {ci_hi:.3f}]",
            "Median ISC": fmt_isc(np.median(x)),
            "SD": fmt_isc(x.std(ddof=1)),
            "Top channel": top_ch,
            "Mean ISC at top channel": fmt_isc(top_val),
        })
    # RS row
    x = rs.isc_grand_mean.dropna().values
    ci_lo, ci_hi = _bootstrap_ci_mean(x)
    rs_ch_cols = [c for c in rs.columns if c.startswith("isc_E") or c == "isc_Cz"]
    chan_mean = rs[rs_ch_cols].apply(pd.to_numeric, errors="coerce").mean(axis=0)
    top_ch_full = chan_mean.idxmax()
    top_ch = top_ch_full.replace("isc_", "")
    top_val = chan_mean.max()
    rows.append({
        "Stimulus": "Resting state (R1)",
        "n": fmt_int(len(x)),
        "Mean ISC (95\\% CI)": f"{x.mean():.3f} [{ci_lo:.3f}, {ci_hi:.3f}]",
        "Median ISC": fmt_isc(np.median(x)),
        "SD": fmt_isc(x.std(ddof=1)),
        "Top channel": top_ch,
        "Mean ISC at top channel": fmt_isc(top_val),
    })
    df = pd.DataFrame(rows)
    cap = ("Table 2. Inter-subject correlation magnitudes per movie. Mean ISC, 95\\% confidence intervals "
           "(bootstrap, 1000 resamples), and top-magnitude channel per movie. Resting-state ISC computed on "
           "R1 RestingState recordings (n=132) as a methodological baseline.")
    write_table(df, 2, "isc_magnitudes", cap, "tab:isc_magnitudes")


# === TABLE 3: age regression ===

def _full_regression(sub):
    formula = ("isc_grand_mean ~ sex_M + age + age_sq + ehq_total + C(release) "
               "+ p_factor + attention + internalizing + externalizing")
    return smf.ols(formula, data=sub).fit()


def table3_age_regression():
    df = _load_analysis_df().dropna(subset=["ehq_total", "p_factor", "attention",
                                             "internalizing", "externalizing"])
    rows = []
    raw_p = []
    for movie in MOVIE_ORDER:
        sub = df[df.movie == movie]
        fit = _full_regression(sub)
        ci = fit.conf_int().loc["age"]
        rows.append({
            "_movie": movie,
            "Movie": MOVIE_LABELS[movie],
            "Age beta": fmt_beta(fit.params["age"]),
            "95\\% CI": fmt_ci(ci[0], ci[1]),
            "t": fmt_t(fit.tvalues["age"]),
            "p": fmt_p(fit.pvalues["age"]),
            "_p": float(fit.pvalues["age"]),
            "R^2": f"{fit.rsquared:.3f}",
            "n": fmt_int(len(sub)),
        })
        raw_p.append(float(fit.pvalues["age"]))
    _, q, _, _ = multipletests(raw_p, method="fdr_bh")
    for i, qv in enumerate(q):
        rows[i]["q FDR"] = fmt_q(qv)
    out_cols = ["Movie", "Age beta", "95\\% CI", "t", "p", "q FDR", "R^2", "n"]
    df_out = pd.DataFrame(rows)[out_cols]
    cap = ("Table 3. Age effect on grand-mean ISC per movie, from the full regression model with sex, age, "
           "age$^2$, EHQ, release, and four CBCL bifactor scores as covariates. Coefficients are per-year "
           "change in ISC. FDR correction applied across four movies.")
    write_table(df_out, 3, "age_regression", cap, "tab:age_regression")


# === TABLE 4: sex regression ===

def table4_sex_regression():
    df = _load_analysis_df()
    df_complete = df.dropna(subset=["ehq_total", "p_factor", "attention",
                                     "internalizing", "externalizing"])
    rows = []
    raw_p_full = []
    for movie in MOVIE_ORDER:
        sub_raw = df[df.movie == movie]
        m = sub_raw[sub_raw.sex == "M"].isc_grand_mean.dropna()
        f = sub_raw[sub_raw.sex == "F"].isc_grand_mean.dropna()
        diff = float(m.mean() - f.mean())
        t_raw, p_raw = ttest_ind(m, f)

        sub_full = df_complete[df_complete.movie == movie]
        fit = _full_regression(sub_full)
        ci = fit.conf_int().loc["sex_M"]
        beta = float(fit.params["sex_M"])
        p_full = float(fit.pvalues["sex_M"])
        rows.append({
            "Movie": MOVIE_LABELS[movie],
            "M-F mean diff": fmt_isc(diff),
            "t (raw)": fmt_t(t_raw),
            "p (raw)": fmt_p(p_raw),
            "Sex_M beta": fmt_beta(beta),
            "95\\% CI": fmt_ci(ci[0], ci[1]),
            "p (regression)": fmt_p(p_full),
            "R^2": f"{fit.rsquared:.3f}",
            "n": fmt_int(len(sub_full)),
        })
        raw_p_full.append(p_full)
    _, q, _, _ = multipletests(raw_p_full, method="fdr_bh")
    for i, qv in enumerate(q):
        rows[i]["q FDR"] = fmt_q(qv)
    out_cols = ["Movie", "M-F mean diff", "t (raw)", "p (raw)",
                "Sex_M beta", "95\\% CI", "p (regression)", "q FDR", "R^2", "n"]
    df_out = pd.DataFrame(rows)[out_cols]
    cap = ("Table 4. Sex effect on grand-mean ISC per movie. Columns 2-4: unadjusted two-sample t-test "
           "(males vs females). Columns 5-9: full regression with controls (sex, age, age$^2$, EHQ, release, "
           "four CBCL scores). FDR correction applied across four movies on the regression-based p-values.")
    write_table(df_out, 4, "sex_regression", cap, "tab:sex_regression")


# === TABLE 5: topographic summary ===

def table5_topography():
    chs = pd.read_csv("outputs/R12345678_sex_effect_per_channel.csv")
    rows = []
    for movie in MOVIE_ORDER:
        sub = chs[chs.movie == movie].copy()
        n_total = len(sub)
        sub_sorted = sub.sort_values("t", ascending=False)
        top_ch = sub_sorted.iloc[0]["channel"]
        max_t = float(sub_sorted.iloc[0]["t"])
        top5 = ", ".join(sub_sorted.head(5)["channel"].tolist())
        sig = sub.q_fdr < 0.05
        n_pos = int((sig & (sub["t"] > 0)).sum())
        n_neg = int((sig & (sub["t"] < 0)).sum())
        pct_pos = 100.0 * n_pos / n_total
        rows.append({
            "Movie": MOVIE_LABELS[movie],
            "Top channel": top_ch,
            "Max t": fmt_t(max_t),
            "Top 5 (by t)": top5,
            "FDR positive (M>F)": fmt_int(n_pos),
            "FDR negative (F>M)": fmt_int(n_neg),
            "Channels tested": fmt_int(n_total),
            "Pct positive": fmt_pct(pct_pos),
        })
    df_out = pd.DataFrame(rows)
    cap = ("Table 5. Topographic distribution of the sex effect on grand-mean ISC per movie. Top channels "
           "by sex effect t-statistic, with the count of channels surviving FDR correction at q $<$ 0.05 "
           "in each direction. The frontocentral cluster (E30, E36, E37) consistently appears among top-5 "
           "channels across all four movies.")
    write_table(df_out, 5, "topography_summary", cap, "tab:topography")


# === TABLE 6: frequency band breakdown ===

def table6_band_breakdown():
    bands = {
        "theta": pd.read_csv("outputs/R12345678_isc_by_band_theta.csv"),
        "alpha": pd.read_csv("outputs/R12345678_isc_by_band_alpha.csv"),
        "beta":  pd.read_csv("outputs/R12345678_isc_by_band_beta.csv"),
    }
    master = pd.read_csv("outputs/R12345678_master.csv")
    sex_lookup = master[["participant_id", "sex"]].drop_duplicates("participant_id")

    rows = []
    for movie in MOVIE_ORDER:
        row = {"Movie": MOVIE_LABELS[movie]}
        ns = []
        for band_name in ["theta", "alpha", "beta"]:
            df_b = bands[band_name].merge(sex_lookup, on="participant_id", how="inner")
            sub = df_b[df_b.movie == movie]
            m = sub[sub.sex == "M"].isc_grand_mean.dropna()
            f = sub[sub.sex == "F"].isc_grand_mean.dropna()
            t, p = ttest_ind(m, f)
            row[f"{band_name.capitalize()} t"] = fmt_t(t)
            row[f"{band_name.capitalize()} p"] = fmt_p(p)
            ns.append(len(m) + len(f))
        row["n"] = fmt_int(int(np.mean(ns)))
        rows.append(row)
    out_cols = ["Movie", "Theta t", "Theta p", "Alpha t", "Alpha p", "Beta t", "Beta p", "n"]
    df_out = pd.DataFrame(rows)[out_cols]
    cap = ("Table 6. Sex effect on band-specific ISC per movie, from the n=400 balanced subsample (200 male, "
           "200 female). Two-sample t-statistics and p-values for the male vs female contrast in theta "
           "(4-8 Hz), alpha (8-12 Hz), and beta (12-20 Hz) ISC. Theta carries the sex effect across all four "
           "movies; alpha and beta show effects only in the abstract control.")
    write_table(df_out, 6, "band_breakdown", cap, "tab:band_breakdown")


# === INDEX ===

def write_index():
    captions = {
        1: ("Cohort demographics by HBN release",
            "table1_demographics",
            "Demographic and clinical characteristics of the analytic cohort by HBN release. "
            "CBCL bifactor scores are standardized scores from the bifactor decomposition of CBCL items."),
        2: ("ISC magnitudes per movie",
            "table2_isc_magnitudes",
            "Inter-subject correlation magnitudes per movie. Mean ISC, 95% confidence intervals "
            "(bootstrap, 1000 resamples), and top-magnitude channel per movie. Resting-state ISC "
            "computed on R1 RestingState recordings (n=132) as a methodological baseline."),
        3: ("Age effect (full regression)",
            "table3_age_regression",
            "Age effect on grand-mean ISC per movie, from the full regression model with sex, age, age², "
            "EHQ, release, and four CBCL bifactor scores as covariates. Coefficients are per-year change "
            "in ISC. FDR correction applied across four movies."),
        4: ("Sex effect (raw + regression)",
            "table4_sex_regression",
            "Sex effect on grand-mean ISC per movie. Columns 2-4: unadjusted two-sample t-test "
            "(males vs females). Columns 5-9: full regression with controls (sex, age, age², EHQ, release, "
            "four CBCL scores). FDR correction applied across four movies on the regression-based p-values."),
        5: ("Topographic summary of the sex effect",
            "table5_topography_summary",
            "Topographic distribution of the sex effect on grand-mean ISC per movie. Top channels by sex "
            "effect t-statistic, with the count of channels surviving FDR correction at q < 0.05 in each "
            "direction. The frontocentral cluster (E30, E36, E37) consistently appears among top-5 channels "
            "across all four movies."),
        6: ("Frequency-band sex effect (n=400 balanced)",
            "table6_band_breakdown",
            "Sex effect on band-specific ISC per movie, from the n=400 balanced subsample (200 male, 200 "
            "female). Two-sample t-statistics and p-values for the male vs female contrast in theta (4-8 Hz), "
            "alpha (8-12 Hz), and beta (12-20 Hz) ISC. Theta carries the sex effect across all four movies; "
            "alpha and beta show effects only in the abstract control."),
    }
    out = ["# Paper tables index", "",
           "Generated 2026-05-06 from analysis outputs at `outputs/`.",
           "Each table is saved as CSV (`tables/csv/`), LaTeX booktabs (`tables/latex/`), and Markdown "
           "(`tables/markdown/`).", ""]
    for n in sorted(captions):
        title, base, cap = captions[n]
        out += [f"## Table {n} - {title}", "",
                f"**Files:** `csv/{base}.csv`, `latex/{base}.tex`, `markdown/{base}.md`", "",
                f"**Caption:** {cap}", ""]
    (TBL_DIR / "INDEX.md").write_text("\n".join(out))
    print(f"  wrote {TBL_DIR / 'INDEX.md'}", flush=True)


def main():
    t0 = time.time()
    print("Building tables...")
    print("\n[Table 1: demographics]")
    table1_demographics()
    print("\n[Table 2: ISC magnitudes]")
    table2_isc_magnitudes()
    print("\n[Table 3: age regression]")
    table3_age_regression()
    print("\n[Table 4: sex regression]")
    table4_sex_regression()
    print("\n[Table 5: topography]")
    table5_topography()
    print("\n[Table 6: band breakdown]")
    table6_band_breakdown()
    print("\n[INDEX]")
    write_index()
    print(f"\nTotal wall: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
