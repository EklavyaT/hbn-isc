"""Gate 2 Tier 2 STEP 5: statistics and the decisive contrasts.

Consumes outputs/gate2_tier2_isc_per_subject.csv (scripts/19). Aggregates per
subject across movies (mean), then computes male vs female contrasts with Welch t
(ttest_ind equal_var=False) and small-sample Hedges g:

  5a primary: frontocentral cluster, DELTA, under A, B, C. g(A), g(B), g(C) and
     the changes g(B) minus g(A), g(C) minus g(A).
  5b secondary: frontocentral cluster, THETA, under A, B, C.
  5c positive control: periocular ring ISC, delta and theta, under A, B, C. The
     manipulation must MOVE periocular ISC from A to B; if not, the test is invalid.
  5d double dissociation: delta, change A to B at frontocentral vs periocular.
  5e ocular-only: sex effect on OC ISC at frontocentral vs periocular, delta and theta.
  5f covariate: regress condition-A frontocentral delta ISC on sex with and
     without the OC frontocentral delta ISC covariate; does the sex term survive.

Print only plus a small contrasts CSV outputs/gate2_tier2_contrasts.csv. Writes no
other file and overwrites nothing.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, t as tdist

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT = PROJECT_ROOT / "outputs"
ISC_CSV = OUT / "gate2_tier2_isc_per_subject.csv"
CONTRASTS_CSV = OUT / "gate2_tier2_contrasts.csv"

CONDS = ["A", "B", "C", "OC"]
BANDS = ["delta", "theta"]
REGIONS = ["fc", "peri", "grand"]


def hedges_g(m, f):
    m = np.asarray(m, float)
    f = np.asarray(f, float)
    nm, nf = len(m), len(f)
    vm, vf = m.var(ddof=1), f.var(ddof=1)
    sp = np.sqrt(((nm - 1) * vm + (nf - 1) * vf) / (nm + nf - 2))
    if sp == 0:
        return 0.0
    d = (m.mean() - f.mean()) / sp
    return d * (1.0 - 3.0 / (4.0 * (nm + nf) - 9.0))


def mvf(per, col):
    m = per[per.sex == "M"][col].dropna()
    f = per[per.sex == "F"][col].dropna()
    t, p = ttest_ind(m, f, equal_var=False)
    return dict(mean_M=float(m.mean()), mean_F=float(f.mean()),
                delta=float(m.mean() - f.mean()), t=float(t), p=float(p),
                g=hedges_g(m.values, f.values), n_M=len(m), n_F=len(f))


def ols_t(y, X, names):
    """Plain OLS with per-coefficient t and p. X includes the intercept column."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    dof = n - k
    sigma2 = (resid @ resid) / dof
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    tvals = beta / se
    pvals = 2 * tdist.sf(np.abs(tvals), dof)
    return {nm: (float(beta[i]), float(se[i]), float(tvals[i]), float(pvals[i]))
            for i, nm in enumerate(names)}


def main():
    if not ISC_CSV.exists():
        print(f"STOP: {ISC_CSV} not found. Run scripts/19 first.")
        sys.exit(1)
    df = pd.read_csv(ISC_CSV)
    val_cols = [f"{c}_{b}_{r}" for c in CONDS for b in BANDS for r in REGIONS]
    # per subject mean across movies
    per = df.groupby(["participant_id", "sex"])[val_cols].mean().reset_index()
    nM = int((per.sex == "M").sum())
    nF = int((per.sex == "F").sum())
    print("=" * 70)
    print(f"GATE 2 TIER 2 STEP 5: stats on {len(per)} subjects (M={nM} F={nF})")
    print("=" * 70)

    rows_out = []

    def emit(tag, region, band, conds=("A", "B", "C")):
        print(f"\n  [{tag}] {region} cluster/ring, {band} band, male vs female:")
        print(f"  {'cond':>4} | {'mean_M':>9} {'mean_F':>9} {'delta':>10} "
              f"{'Welch t':>9} {'p':>11} {'g':>8}")
        res = {}
        for c in conds:
            r = mvf(per, f"{c}_{band}_{region}")
            res[c] = r
            rows_out.append({"contrast": tag, "region": region, "band": band,
                             "cond": c, **{k: r[k] for k in
                             ("mean_M", "mean_F", "delta", "t", "p", "g")}})
            print(f"  {c:>4} | {r['mean_M']:9.5f} {r['mean_F']:9.5f} {r['delta']:+10.5f} "
                  f"{r['t']:9.3f} {r['p']:11.3e} {r['g']:8.3f}")
        return res

    # 5a primary
    print("\n" + "-" * 70)
    print("  5a PRIMARY ENDPOINT: frontocentral, DELTA, under A, B, C")
    print("-" * 70)
    a = emit("5a", "fc", "delta")
    print(f"\n  g(A)={a['A']['g']:.3f} g(B)={a['B']['g']:.3f} g(C)={a['C']['g']:.3f} | "
          f"g(B)-g(A)={a['B']['g']-a['A']['g']:+.3f} g(C)-g(A)={a['C']['g']-a['A']['g']:+.3f}")

    # 5b secondary
    print("\n" + "-" * 70)
    print("  5b SECONDARY: frontocentral, THETA, under A, B, C")
    print("-" * 70)
    b = emit("5b", "fc", "theta")
    print(f"\n  g(A)={b['A']['g']:.3f} g(B)={b['B']['g']:.3f} g(C)={b['C']['g']:.3f} | "
          f"g(B)-g(A)={b['B']['g']-b['A']['g']:+.3f} g(C)-g(A)={b['C']['g']-b['A']['g']:+.3f}")

    # 5c positive control
    print("\n" + "-" * 70)
    print("  5c POSITIVE CONTROL: periocular ring ISC, delta and theta, A/B/C")
    print("  (the manipulation must move periocular ISC from A to B)")
    print("-" * 70)
    for band in BANDS:
        emit("5c", "peri", band)
        pa = per[f"A_{band}_peri"].mean()
        pb = per[f"B_{band}_peri"].mean()
        pc = per[f"C_{band}_peri"].mean()
        print(f"  periocular pooled mean ISC {band}: A={pa:.5f} B={pb:.5f} C={pc:.5f} "
              f"| B-A={pb-pa:+.5f} ({(pb-pa)/abs(pa)*100:+.1f}% of A)")

    # 5d double dissociation, delta
    print("\n" + "-" * 70)
    print("  5d DOUBLE DISSOCIATION (delta): change A to B at frontocentral vs periocular")
    print("-" * 70)
    for region, lab in [("fc", "frontocentral"), ("peri", "periocular")]:
        va = per[f"A_delta_{region}"].mean()
        vb = per[f"B_delta_{region}"].mean()
        print(f"  {lab:>14}: A={va:.5f} B={vb:.5f} | B-A={vb-va:+.5f} "
              f"({(vb-va)/abs(va)*100:+.1f}% of A)")
    fc_chg = per["B_delta_fc"].mean() - per["A_delta_fc"].mean()
    peri_chg = per["B_delta_peri"].mean() - per["A_delta_peri"].mean()
    ratio = abs(peri_chg) / abs(fc_chg) if fc_chg != 0 else np.inf
    print(f"  periocular change is {ratio:.1f}x the frontocentral change. "
          f"Dissociation holds: {abs(peri_chg) > 3 * abs(fc_chg)}")

    # 5e ocular-only contribution
    print("\n" + "-" * 70)
    print("  5e OCULAR-ONLY (OC) sex effect: frontocentral vs periocular, delta and theta")
    print("-" * 70)
    for band in BANDS:
        rfc = mvf(per, f"OC_{band}_fc")
        rpe = mvf(per, f"OC_{band}_peri")
        rows_out.append({"contrast": "5e", "region": "fc", "band": band, "cond": "OC",
                         **{k: rfc[k] for k in ("mean_M", "mean_F", "delta", "t", "p", "g")}})
        rows_out.append({"contrast": "5e", "region": "peri", "band": band, "cond": "OC",
                         **{k: rpe[k] for k in ("mean_M", "mean_F", "delta", "t", "p", "g")}})
        print(f"  {band}: OC frontocentral g={rfc['g']:+.3f} (t={rfc['t']:.2f}, "
              f"p={rfc['p']:.2e}); OC periocular g={rpe['g']:+.3f} "
              f"(t={rpe['t']:.2f}, p={rpe['p']:.2e})")

    # 5f covariate regression
    print("\n" + "-" * 70)
    print("  5f COVARIATE: A frontocentral delta ISC ~ sex (+ OC frontocentral delta ISC)")
    print("-" * 70)
    d = per.dropna(subset=["A_delta_fc", "OC_delta_fc"]).copy()
    sex01 = (d.sex == "M").astype(float).values
    y = d["A_delta_fc"].values
    oc = d["OC_delta_fc"].values
    n = len(d)
    m1 = ols_t(y, np.column_stack([np.ones(n), sex01]), ["intercept", "sex"])
    m2 = ols_t(y, np.column_stack([np.ones(n), sex01, oc]),
               ["intercept", "sex", "OC_delta_fc"])
    bs1, se1, t1, p1 = m1["sex"]
    bs2, se2, t2, p2 = m2["sex"]
    bo, seo, to, po = m2["OC_delta_fc"]
    print(f"  model 1 (sex only)     : sex beta={bs1:+.5f} (se {se1:.5f}) t={t1:.2f} p={p1:.3e}")
    print(f"  model 2 (sex + OC cov) : sex beta={bs2:+.5f} (se {se2:.5f}) t={t2:.2f} p={p2:.3e}")
    print(f"                           OC  beta={bo:+.5f} (se {seo:.5f}) t={to:.2f} p={po:.3e}")
    survive = (p2 < 0.05) and (np.sign(bs2) == np.sign(bs1))
    retain = bs2 / bs1 * 100 if bs1 != 0 else np.nan
    print(f"  sex term survives OC covariate: {survive} "
          f"(retains {retain:.1f}% of its size)")
    rows_out.append({"contrast": "5f", "region": "fc", "band": "delta", "cond": "modelA_sexonly",
                     "delta": bs1, "t": t1, "p": p1, "g": np.nan,
                     "mean_M": np.nan, "mean_F": np.nan})
    rows_out.append({"contrast": "5f", "region": "fc", "band": "delta", "cond": "modelB_sex_plus_OC",
                     "delta": bs2, "t": t2, "p": p2, "g": np.nan,
                     "mean_M": np.nan, "mean_F": np.nan})

    # ---- VERDICT ----
    print("\n" + "=" * 70)
    print("  VERDICT (Tier 2)")
    print("=" * 70)
    g_a, g_b, g_c = a["A"]["g"], a["B"]["g"], a["C"]["g"]
    fc_stable = max(abs(g_b - g_a), abs(g_c - g_a)) < 0.20 * abs(g_a)
    peri_moves = abs(peri_chg) > 0.10 * abs(per["A_delta_peri"].mean())
    dissoc = abs(peri_chg) > 3 * abs(fc_chg)
    oc_fc_small = abs(mvf(per, "OC_delta_fc")["g"]) < 0.5 * abs(g_a)
    print(f"  frontocentral delta effect stable across A/B/C (g {g_a:.2f}/{g_b:.2f}/{g_c:.2f}): {fc_stable}")
    print(f"  periocular positive control moved A to B: {peri_moves}")
    print(f"  double dissociation (periocular moves, frontocentral does not): {dissoc}")
    print(f"  ocular-only sex effect at frontocentral small vs the cluster effect: {oc_fc_small}")
    print(f"  sex term survives OC covariate: {survive}")
    exoneration = fc_stable and peri_moves and dissoc and oc_fc_small and survive
    if exoneration:
        print("  => EXONERATION: ocular components do not reach the cluster and do not "
              "carry the effect.")
    else:
        print("  => NOT a clean exoneration. Report which endpoints passed and which "
              "did not (see above); do not force a verdict.")

    pd.DataFrame(rows_out).to_csv(CONTRASTS_CSV, index=False)
    print(f"\n  wrote {CONTRASTS_CSV}")
    print("STEP 5 COMPLETE")


if __name__ == "__main__":
    main()
