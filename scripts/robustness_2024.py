#!/usr/bin/env python
# coding: utf-8

# %% [markdown]
# # Robustness check: replication with the 2024 Citywide Mobility Survey
#
# The 2024 CMS does not ask about the employer's in-office requirement (`wfh_policy`), so WFHA
# cannot be rebuilt for 2024. What both waves share is the respondent's **actual telework
# frequency**, the measure used for the "actual telework frequency" row of the robustness grid.
# This script estimates the same models with that measure in 2022 and 2024 and compares them.
#
# | Section | Content |
# |---|---|
# | 1 | Samples and the harmonized telework scale (T, 0–6) |
# | 2 | Main association: linear, quadratic, + Manhattan, + borough and industry, weighted |
# | 3 | Three-level contrasts (none / up to 1 day a week / 2+ days): logit, LPM, weighted, Firth |
# | 4 | Robustness grid (joint p of the telework terms; pooled and four groups) |
# | 5 | Does the association differ between 2022 and 2024? |
# | 6 | Heterogeneity by age and workplace location |
# | 7 | Mediation through household vehicles |
# | 8 | Weekly car-commute days and commute days |
# | 9 | Reported change in household vehicles |
#
# **Inputs** (NYC Open Data, exported as CSV; see the README):
# `data/raw/Citywide_Mobility_Survey_-_Person_HOUSEHOLD.csv` (2022 person table),
# `data/raw/Citywide_Mobility_Survey_-_Person_2024.csv`, `data/raw/Citywide_Mobility_Survey_-_Household_2024.csv`.
#
# The 2022 sample uses the corrected sample definition of the revised manuscript (all commuters
# with a valid employer policy, including daily commuters; "other" mode excluded), not the
# v4 definition in `replication.py`. No values are checked against the manuscript, because
# these results are not yet reported there.

# %%
import json
import os
import warnings

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2

warnings.filterwarnings("ignore")
pd.set_option("display.width", 160)

ROOT = os.path.abspath("..") if os.path.basename(os.getcwd()) in ("notebooks", "scripts") else os.getcwd()
RAW = os.path.join(ROOT, "data", "raw")
CSV22 = os.environ.get("WFH_CSV", os.path.join(RAW, "Citywide_Mobility_Survey_-_Person_HOUSEHOLD.csv"))
CSV24P = os.environ.get("WFH_CSV_2024_PERSON", os.path.join(RAW, "Citywide_Mobility_Survey_-_Person_2024.csv"))
CSV24H = os.environ.get("WFH_CSV_2024_HOUSEHOLD", os.path.join(RAW, "Citywide_Mobility_Survey_-_Household_2024.csv"))
RESULTSDIR = os.environ.get("WFH_RESULTSDIR", os.path.join(ROOT, "results"))
os.makedirs(RESULTSDIR, exist_ok=True)
B_BOOT = int(os.environ.get("WFH_BOOT", 1000))
RNG = np.random.default_rng(2024)
RESULTS = {}

SUST = {1, 102, 103, 105, 107, 108}   # walk, bus, bicycle, rail, micromobility, ferry
NONS = {100, 101, 106}                 # household vehicle, other vehicle, taxi / ride-hailing
CAR = NONS
# Telework harmonized to the 2022 recoded scale: 0 = never (incl. daily commuters) ... 6 = 5+ days a week.
# 2022 telework_freq: 1 = 5+ days, 2 = 4, 3 = 2-3, 4 = 1 day/week, 5 = 1-3 days/month, 6 = < monthly, 996 = never.
# 2024 telework_freq: 1 = 5+ days, 2 = 4, 3 = 3, 4 = 2, 5 = 1 day/week, 6 = 1-3 days/month, 7 = < monthly, 996 = never.
TW22 = {996: 0, 6: 1, 5: 2, 4: 3, 3: 4, 2: 5, 1: 6}
TW24 = {996: 0, 7: 1, 6: 2, 5: 3, 4: 4, 3: 4, 2: 5, 1: 6}
DAYS22 = {1: 5, 2: 4, 3: 2.5, 4: 1, 5: 0.46, 6: 0.12}
DAYS24 = {1: 5, 2: 4, 3: 3, 4: 2, 5: 1, 6: 0.46, 7: 0.12}
WFH_POLICY = {1, 2, 3, 4, 5, 6, 996}
NYC = {36005, 36047, 36061, 36081, 36085}
# Knowledge-sector industries (finance, real estate, professional services, government, non-profit,
# technology, media). The 2024 list has no non-profit category.
KNOW22 = {1, 2, 4, 14, 15, 16, 17}
KNOW24 = {1, 7, 8, 9, 10, 11}

CT = "agec + inc + edu + cross"
Q = "T + I(T**2)"


def num(x):
    return pd.to_numeric(x.astype(str).str.replace(",", ""), errors="coerce")


def collapse_industry(s, minn=15):
    ind = s.industry.fillna(997).astype(int).astype(str).replace("995", "997")
    vc = ind.value_counts()
    ind = ind.where(~ind.isin(vc[vc < minn].index), "oth").replace("997", "oth")
    g = s.groupby(ind).Y.mean()
    return ind.where(~ind.isin(g[(g == 0) | (g == 1)].index), "oth")


def finish(s, year):
    s = s.copy()
    s["year"] = year
    s["Y"] = s.work_mode.isin(SUST).astype(int)
    s["car"] = s.work_mode.isin(CAR).astype(int)
    s["cross"] = (s.home_county != s.work_county).astype(int)
    # Manhattan workplace: CMS work zones 3-4 in 2022, as in the manuscript; 2024 renumbered its zones, so use New York County
    s["manh"] = (s.work_cms_zone.isin([3, 4]) if year == 2022 else s.work_county == 36061).astype(int)
    s["veh"] = s.num_vehicles.clip(upper=2)
    s["anyveh"] = (s.num_vehicles > 0).astype(int)
    s["inc"], s["edu"], s["agec"] = s.income_broad, s.education, s.age
    s["young"] = (s.agec <= 6).astype(int)
    s["hboro"] = s.home_county.astype(int).astype(str)
    s["w"] = s.person_weight / s.person_weight.mean()
    s["tg"] = pd.cut(s["T"], [-1, 0, 3, 6], labels=["none", "occasional", "regular"])
    s["cardays"] = s.days * s.car
    s["parttime"] = (s.employment == 2).astype(int)
    s["woman"] = (s.gender == 1).astype(int)
    s["kids"] = (s.num_kids > 0).astype(int)
    s["outside"] = (~s.work_county.isin(NYC)).astype(int)
    s["ind"] = collapse_industry(s)
    return s.reset_index(drop=True)


# %% [markdown]
# ## 1. Samples and the harmonized telework scale

# %%
d22 = pd.read_csv(CSV22, low_memory=False)
m = (d22.wfh_policy.isin(WFH_POLICY) & d22.work_mode.isin(SUST | NONS) & d22.age.between(4, 11)
     & d22.education.between(1, 7) & d22.income_broad.between(1, 8)
     & d22.work_county.notna() & (d22.work_county != 995))
a = d22[m].copy()
a["T"] = a.r_telework_freq.map(TW22)
a["days"] = a.commute_freq.map(DAYS22)
a = finish(a[a["T"].notna()], 2022)
a["know"] = a.industry.isin(KNOW22).astype(int)

p24 = pd.read_csv(CSV24P, low_memory=False)
h24 = pd.read_csv(CSV24H, low_memory=False)
d24 = p24.merge(h24[["hh_id", "home_county", "income_broad", "num_vehicles", "num_kids", "vehicle_change", "reduced_veh_3"]],
                on="hh_id", how="left")
d24["work_county"] = num(d24.work_county)
d24["person_weight"] = num(d24.person_weight_citywide)
# As in 2022, telework_freq was asked only of respondents who do not commute 5+ days a week.
d24["r_telework_freq"] = np.where(d24.commute_freq == 1, 996, d24.telework_freq)
flow = [("All person records", len(d24))]
m = d24.work_mode.isin(SUST | NONS); flow.append(('Valid commute mode, excluding "other"', int(m.sum())))
m &= d24.age.between(4, 11); flow.append(("Adult", int(m.sum())))
m &= d24.education.between(1, 7); flow.append(("Valid education", int(m.sum())))
m &= d24.income_broad.between(1, 8); flow.append(("Valid income", int(m.sum())))
m &= d24.work_county.notna(); flow.append(("Valid work county", int(m.sum())))
m &= d24.commute_freq.between(1, 7) & d24.r_telework_freq.isin(TW24); flow.append(("Valid commute and telework frequency", int(m.sum())))
b = d24[m].copy()
b["T"] = b.r_telework_freq.map(TW24)
b["days"] = b.commute_freq.map(DAYS24)
b = finish(b, 2024)
b["know"] = b.industry.isin(KNOW24).astype(int)
YEARS = {2022: a, 2024: b}

RESULTS["sample"] = {"flow_2024": flow, "n": {y: len(s) for y, s in YEARS.items()},
                     "household_member_1_share": {y: float((s.person_num == 1).mean()) for y, s in YEARS.items()}}
for y, s in YEARS.items():
    print(f"\n{y}: N = {len(s)}; sustainable share {s.Y.mean():.3f}; Manhattan workplace {s.manh.mean():.3f}; "
          f"daily commuters {(s.days == 5).mean():.3f}; Kish ESS {s.w.sum()**2/(s.w**2).sum():.0f}")
print("\n2024 sample flow:"); [print(f"  {k:45s} {v:,}") for k, v in flow]
tab = pd.concat({y: s.groupby("T").Y.agg(["size", "mean"]) for y, s in YEARS.items()}, axis=1).round(3)
print("\nSustainable share by telework level T:\n", tab)
RESULTS["share_by_T"] = {y: s.groupby("T").Y.agg(["size", "mean"]).reset_index().values.tolist() for y, s in YEARS.items()}

# %% [markdown]
# ## 2. Main association (Table 2 analogue)

# %%
def logit(f, s, cov="HC3"):
    return smf.logit(f, s).fit(disp=0, cov_type=cov, maxiter=300)


def wlogit(f, s):
    return smf.glm(f, s, family=sm.families.Binomial(), var_weights=s.w).fit(cov_type="HC1")


def wald(res, pattern):
    """Joint Wald p. pattern "T" means exactly the telework terms; otherwise a substring match (e.g. "tg")."""
    names = [n for n in res.params.index if (n in ("T", "I(T ** 2)") if pattern == "T" else pattern in n)]
    R = np.zeros((len(names), len(res.params)))
    for i, n in enumerate(names):
        R[i, list(res.params.index).index(n)] = 1
    t = res.wald_test(R, scalar=True)
    return float(t.pvalue)


def ame_q(res, s, weights=None):
    b1 = res.params["T"]; b2 = res.params.get("I(T ** 2)", 0.0)
    p = res.predict(s)
    return float(np.average(p * (1 - p) * (b1 + 2 * b2 * s["T"]), weights=weights))


def boot(fun, s, n=B_BOOT):
    out = []
    for _ in range(n):
        df = s.sample(len(s), replace=True, random_state=int(RNG.integers(1e9))).reset_index(drop=True)
        try:
            out.append(fun(df))
        except Exception:
            pass
    return np.array(out)


def ci(x):
    return np.percentile(x, [2.5, 97.5]).tolist()


SPECS = {"(1) Linear": f"Y ~ T + {CT}",
         "(2) Quadratic": f"Y ~ {Q} + {CT}",
         "(3) + Manhattan": f"Y ~ {Q} + manh + {CT}",
         "(4) + borough, industry": f"Y ~ {Q} + manh + C(hboro) + C(ind) + {CT}"}
main = {}
for y, s in YEARS.items():
    rows = {}
    for k, f in SPECS.items():
        r = logit(f, s)
        row = {"b_T": r.params["T"], "se_T": r.bse["T"], "p_T": r.pvalues["T"], "pr2": r.prsquared}
        if "I(T ** 2)" in r.params:
            row.update(b_T2=r.params["I(T ** 2)"], se_T2=r.bse["I(T ** 2)"], p_T2=r.pvalues["I(T ** 2)"], joint=wald(r, "T"))
            lin = smf.logit(f.replace(" + I(T**2)", ""), s).fit(disp=0); q = smf.logit(f, s).fit(disp=0)
            lr = 2 * (q.llf - lin.llf); row["lr_quad"] = (lr, chi2.sf(lr, 1))
            bb = boot(lambda df: ame_q(smf.logit(f, df).fit(disp=0), df), s, 500)
            row["ame"] = (ame_q(r, s), *ci(bb))
            if k == "(2) Quadratic":
                row["vertex"] = -r.params["T"] / (2 * r.params["I(T ** 2)"])
                vb = boot(lambda df: (lambda p: -p["T"] / (2 * p["I(T ** 2)"]))(smf.logit(f, df).fit(disp=0).params), s)
                row["vertex_ci"] = ci(vb)
        rows[k] = row
    f = SPECS["(4) + borough, industry"]
    r = wlogit(f, s)
    bb = boot(lambda df: ame_q(wlogit(f, df), df, df.w), s, 500)
    rows["(5) Weighted (4)"] = {"b_T": r.params["T"], "se_T": r.bse["T"], "b_T2": r.params["I(T ** 2)"], "se_T2": r.bse["I(T ** 2)"],
                                "joint": wald(r, "T"), "ame": (ame_q(r, s, s.w), *ci(bb))}
    main[y] = rows
RESULTS["main"] = main
for y, rows in main.items():
    print(f"\n{y}  (logit, HC3; weighted column HC1)")
    for k, r in rows.items():
        q = f"T² {r['b_T2']: .3f} ({r['se_T2']:.3f})" if "b_T2" in r else ""
        jt = f"joint p {r['joint']:.4f}" if "joint" in r else f"p {r['p_T']:.4f}"
        am = f"AME {r['ame'][0]:.3f} [{r['ame'][1]:.3f}, {r['ame'][2]:.3f}]" if "ame" in r else ""
        lr = f"LR quad p {r['lr_quad'][1]:.3f}" if "lr_quad" in r else ""
        print(f"  {k:26s} T {r['b_T']: .3f} ({r['se_T']:.3f})  {q:22s} {jt:15s} {am:30s} {lr}")
    v = rows["(2) Quadratic"]
    print(f"  vertex {v['vertex']:.2f} [{v['vertex_ci'][0]:.2f}, {v['vertex_ci'][1]:.2f}]")

# %% [markdown]
# ## 3. Three-level contrasts (none / occasional, up to 1 day a week / regular, 2+ days a week)

# %%
def _pred(res, df, beta):
    X = patsy.dmatrix(res.model.data.design_info, df, return_type="dataframe")
    eta = X.values @ beta
    if isinstance(res.model, sm.GLM):
        return res.model.family.link.inverse(eta)
    if hasattr(res.model, "cdf"):
        return res.model.cdf(eta)
    return eta


def contrast(res, s, a, b, weighted=False):
    """Standardized prediction difference E[y | tg = a] - E[y | tg = b] with a delta-method SE."""
    da, db = s.copy(), s.copy()
    da["tg"] = pd.Categorical([a] * len(s), categories=s.tg.cat.categories)
    db["tg"] = pd.Categorical([b] * len(s), categories=s.tg.cat.categories)
    w = s.w.values if weighted else np.ones(len(s))
    beta = res.params.values; V = res.cov_params().values
    f = lambda bb: np.average(_pred(res, da, bb) - _pred(res, db, bb), weights=w)
    g = np.array([(f(beta + e) - f(beta - e)) / 2e-6 for e in np.eye(len(beta)) * 1e-6])
    return float(f(beta)), float(np.sqrt(g @ V @ g))


def firth(y, X, iters=100):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b)); W = p * (1 - p)
        I = X.T @ (X * W[:, None]); Ii = np.linalg.inv(I)
        h = np.einsum("ij,jk,ik->i", X * np.sqrt(W)[:, None], Ii, X * np.sqrt(W)[:, None])
        step = np.clip(Ii @ (X.T @ (y - p + h * (0.5 - p))), -2, 2); b = b + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return b


def firth_lr(y, X, drop):
    def pll(Xm):
        bb = firth(y, Xm); p = 1 / (1 + np.exp(-Xm @ bb))
        return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)) + 0.5 * np.linalg.slogdet(Xm.T @ (Xm * (p * (1 - p))[:, None]))[1]
    keep = [i for i in range(X.shape[1]) if i not in drop]
    lr = 2 * (pll(X) - pll(X[:, keep])); return float(chi2.sf(lr, len(drop)))


BASE3 = "C(tg, Treatment('none')) + C(age4) + inc + edu + cross"
FULL3 = BASE3 + " + manh + C(hboro) + C(ind)"
cat = {}
for y, s in YEARS.items():
    s["age4"] = pd.cut(s.agec, [0, 5, 6, 7, 11], labels=["18-34", "35-44", "45-54", "55+"])
    for lab, f in [("demographic", BASE3), ("full", FULL3)]:
        for wt in (False, True):
            r = wlogit("Y ~ " + f, s) if wt else smf.logit("Y ~ " + f, s).fit(disp=0, cov_type="HC1", maxiter=300)
            cat[f"{y}|{lab}|{'weighted' if wt else 'unweighted'}|logit"] = dict(
                occasional=contrast(r, s, "occasional", "none", wt), regular=contrast(r, s, "regular", "none", wt), p=wald(r, "tg"))
        r = smf.ols("Y ~ " + f, s).fit(cov_type="HC1")
        nm = lambda lv: f"C(tg, Treatment('none'))[T.{lv}]"
        cat[f"{y}|{lab}|unweighted|lpm"] = dict(occasional=(r.params[nm("occasional")], r.bse[nm("occasional")]),
                                                regular=(r.params[nm("regular")], r.bse[nm("regular")]), p=wald(r, "tg"))
        X = patsy.dmatrix(f, s, return_type="dataframe")
        cat[f"{y}|{lab}|firth_p"] = firth_lr(s.Y.values.astype(float), X.values, [i for i, c in enumerate(X.columns) if "tg" in c])
RESULTS["categorical"] = cat
print("\nPercentage-point differences from no telework [95% CI]; joint p")
for k, v in cat.items():
    if k.endswith("firth_p"):
        print(f"  {k:40s} Firth LR p {v:.4f}"); continue
    fm = lambda e: f"{100*e[0]:5.1f} [{100*(e[0]-1.96*e[1]):5.1f}, {100*(e[0]+1.96*e[1]):5.1f}]"
    print(f"  {k:40s} occasional {fm(v['occasional'])}  regular {fm(v['regular'])}  p {v['p']:.4f}")

# %% [markdown]
# ## 4. Robustness grid (joint Wald p of the telework terms)
#
# Columns: pooled (with age), younger (18–44), older (45+), Manhattan workplace, elsewhere (groups without age).

# %%
GROUPS = [("pooled", None, None), ("younger", "young", 1), ("older", "young", 0), ("Manhattan", "manh", 1), ("elsewhere", "manh", 0)]


def grid_row(s0, rhs="T + I(T**2)", extra="", pat="T", test=wald):
    out = []
    for g, var, val in GROUPS:
        s = s0 if var is None else s0[s0[var] == val]
        s = s.copy(); s["ind"] = collapse_industry(s) if "ind" in extra else s.get("ind")
        ctl = (CT if var is None else "inc + edu + cross") + ("" if var == "manh" else extra) + (extra.replace("+ manh", "") if var == "manh" else "")
        try:
            out.append(test(logit(f"Y ~ {rhs} + {ctl}", s), pat))
        except Exception:
            out.append(np.nan)
    return out


def pval(name):
    return lambda res, pat: float(res.pvalues[name])


grid = {}
for y, s in YEARS.items():
    s = s.copy(); s["weekly"] = (s["T"] >= 3).astype(int); s["regular"] = (s["T"] >= 4).astype(int)
    G = {}
    G["Primary: T (0–6), quadratic"] = grid_row(s)
    G["T, linear only"] = grid_row(s, "T")
    G["Three levels (none / occasional / regular)"] = grid_row(s, "C(tg, Treatment('none'))", pat="tg")
    G["Weekly telework (T ≥ 3) vs. less"] = grid_row(s, "weekly", test=pval("weekly"))
    G["Regular telework (2+ days, T ≥ 4) vs. less"] = grid_row(s, "regular", test=pval("regular"))
    G["+ Manhattan workplace"] = grid_row(s, extra=" + manh")
    G["+ home borough"] = grid_row(s, extra=" + C(hboro)")
    G["+ home borough, gender, children, part-time"] = grid_row(s, extra=" + C(hboro) + woman + kids + parttime")
    G["+ home borough, industry"] = grid_row(s, extra=" + C(hboro) + C(ind)")
    G["+ Manhattan, home borough, industry"] = grid_row(s, extra=" + manh + C(hboro) + C(ind)")
    G["+ household vehicles (direct association)"] = grid_row(s, extra=" + veh")
    G["Full-time employees only"] = grid_row(s[s.employment == 1])
    G["Excluding 5+ telework days (T = 6)"] = grid_row(s[s["T"] < 6])
    G["Excluding workplaces outside NYC"] = grid_row(s[s.outside == 0])
    G["Non-daily commuters only"] = grid_row(s[s.days < 5])
    G["Knowledge-sector industries only"] = grid_row(s[s.know == 1])
    if y == 2024:
        s7 = s.copy(); s7["T"] = s7.r_telework_freq.map({996: 0, 7: 1, 6: 2, 5: 3, 4: 4, 3: 5, 2: 6, 1: 7})
        G["Finer 2024 scale (0–7; 2 and 3 days separate)"] = grid_row(s7)
    n = {"Full-time employees only": int((s.employment == 1).sum()), "Excluding 5+ telework days (T = 6)": int((s["T"] < 6).sum()),
         "Excluding workplaces outside NYC": int((s.outside == 0).sum()), "Non-daily commuters only": int((s.days < 5).sum()),
         "Knowledge-sector industries only": int(s.know.sum())}
    grid[y] = {"rows": G, "n": n}
RESULTS["grid"] = grid
for y in YEARS:
    print(f"\n{y} robustness grid: joint p (pooled, younger, older, Manhattan, elsewhere)")
    for k, row in grid[y]["rows"].items():
        print(f"  {k:52s} " + "  ".join("   —  " if pd.isna(v) else f"{v:6.4f}" for v in row))

# %% [markdown]
# ## 5. Does the association differ between 2022 and 2024?

# %%
Z = pd.concat([a, b], ignore_index=True)
Z["y24"] = (Z.year == 2024).astype(int)
Z["ind"] = Z.year.astype(str) + "_" + Z["ind"].astype(str)
yr = {}
for lab, extra in [("demographic", ""), ("+ Manhattan", " + manh"), ("full", " + manh + C(hboro) + C(ind)")]:
    f0 = f"Y ~ {Q} + {CT} + y24{extra}"
    r0 = smf.logit(f0, Z).fit(disp=0, maxiter=300); r1 = smf.logit(f0 + " + T:y24 + I(T**2):y24", Z).fit(disp=0, maxiter=300)
    lr = 2 * (r1.llf - r0.llf); yr[lab] = (lr, 2, chi2.sf(lr, 2))
f = "Y ~ C(tg, Treatment('none')) + C(age4) + inc + edu + cross + manh + y24"
Z["age4"] = pd.cut(Z.agec, [0, 5, 6, 7, 11], labels=["18-34", "35-44", "45-54", "55+"])
r0 = smf.logit(f, Z).fit(disp=0); r1 = smf.logit(f + " + C(tg, Treatment('none')):y24", Z).fit(disp=0)
lr = 2 * (r1.llf - r0.llf); yr["three levels, + Manhattan"] = (lr, 2, chi2.sf(lr, 2))
RESULTS["year_interaction"] = yr
print("\nTelework × year interaction (LR tests, pooled 2022 + 2024):")
for k, v in yr.items():
    print(f"  {k:28s} chi2({v[1]}) = {v[0]:.2f}, p = {v[2]:.3f}")

# %% [markdown]
# ## 6. Heterogeneity by age and workplace location

# %%
het = {}
for y, s in YEARS.items():
    groups = {}
    for nm, var, val in GROUPS[1:]:
        g = s[s[var] == val].reset_index(drop=True); f = f"Y ~ {Q} + inc + edu + cross"
        r = logit(f, g); bb = boot(lambda df: ame_q(smf.logit(f, df).fit(disp=0), df), g)
        groups[nm] = dict(n=len(g), share=g.Y.mean(), b_T=r.params["T"], p_T=r.pvalues["T"], b_T2=r.params["I(T ** 2)"],
                          p_T2=r.pvalues["I(T ** 2)"], joint=wald(r, "T"), ame=(ame_q(r, g), *ci(bb)))
    tests = {}
    for lab, base, term in [("age", f"Y ~ {Q} + inc + edu + cross + young", "T:young + I(T**2):young"),
                            ("Manhattan", f"Y ~ {Q} + {CT} + manh", "T:manh + I(T**2):manh"),
                            ("age, full controls", f"Y ~ {Q} + inc + edu + cross + young + manh + C(hboro) + C(ind)", "T:young + I(T**2):young"),
                            ("Manhattan, full controls", f"Y ~ {Q} + {CT} + manh + C(hboro) + C(ind)", "T:manh + I(T**2):manh")]:
        r0 = smf.logit(base, s).fit(disp=0, maxiter=300); r1 = smf.logit(base + " + " + term, s).fit(disp=0, maxiter=300)
        lr = 2 * (r1.llf - r0.llf); tests[lab] = (lr, 2, chi2.sf(lr, 2))
    het[y] = {"groups": groups, "interactions": tests}
RESULTS["heterogeneity"] = het
for y, h in het.items():
    print(f"\n{y} groups (quadratic, no age term):")
    for k, v in h["groups"].items():
        print(f"  {k:10s} n {v['n']:5d} share {v['share']:.3f}  joint p {v['joint']:.4f}  T² {v['b_T2']: .3f} (p {v['p_T2']:.3f})  "
              f"AME {v['ame'][0]:.3f} [{v['ame'][1]:.3f}, {v['ame'][2]:.3f}]")
    for k, v in h["interactions"].items():
        print(f"  interaction × {k:26s} chi2(2) = {v[0]:.2f}, p = {v[2]:.3f}")

# %% [markdown]
# ## 7. Mediation through household vehicles (Table 4 analogue; exploratory)

# %%
def med(df, ctl):
    t = smf.logit(f"Y ~ {Q} + {ctl}", df).fit(disp=0); dr = smf.logit(f"Y ~ {Q} + veh + {ctl}", df).fit(disp=0)
    return ame_q(t, df), ame_q(dr, df)


mt = {}
for y, s in YEARS.items():
    rows = {}
    for nm, var, val in [("All workers", None, None)] + GROUPS[1:]:
        g = (s if var is None else s[s[var] == val]).reset_index(drop=True); ctl = CT if var is None else "inc + edu + cross"
        pa = smf.ols(f"veh ~ T + {ctl}", g).fit(cov_type="HC3"); t, dr = med(g, ctl)
        bb = boot(lambda df: (lambda x: (x[0], x[1], x[0] - x[1]))(med(df, ctl)), g)
        rows[nm] = dict(n=len(g), path_a=(pa.params["T"], pa.pvalues["T"]), total=(t, *ci(bb[:, 0])), direct=(dr, *ci(bb[:, 1])),
                        indirect=(t - dr, *ci(bb[:, 2])))
    t0 = ame_q(smf.logit(f"Y ~ {Q} + C(hboro) + {CT}", s).fit(disp=0), s)
    t1 = ame_q(smf.logit(f"Y ~ {Q} + veh + C(hboro) + {CT}", s).fit(disp=0), s)
    rows["share with home borough"] = 1 - t1 / t0
    own = s[s.anyveh == 1].copy(); own["ind"] = collapse_industry(own); own = own.reset_index(drop=True)
    r = smf.logit("Y ~ " + FULL3, own).fit(disp=0, cov_type="HC1", maxiter=300)
    rows["car-owning households, full controls"] = dict(n=len(own), occasional=contrast(r, own, "occasional", "none"),
                                                         regular=contrast(r, own, "regular", "none"), p=wald(r, "tg"))
    mt[y] = rows
RESULTS["mediation"] = mt
for y, rows in mt.items():
    print(f"\n{y} mediation (AMEs; percentile bootstrap, {B_BOOT} resamples)")
    for k, v in rows.items():
        if k.startswith("All") or k in ("younger", "older", "Manhattan", "elsewhere"):
            sh = f"{100*v['indirect'][0]/v['total'][0]:.0f}%" if v["total"][0] > 0 else "—"
            print(f"  {k:12s} n {v['n']:5d} path a {v['path_a'][0]: .3f} (p {v['path_a'][1]:.4f})  total {v['total'][0]:.3f} "
                  f"[{v['total'][1]:.3f}, {v['total'][2]:.3f}]  indirect {v['indirect'][0]: .3f} [{v['indirect'][1]: .3f}, {v['indirect'][2]: .3f}]  {sh}")
    print(f"  indirect share with home borough: {100*rows['share with home borough']:.0f}%")
    o = rows["car-owning households, full controls"]
    print(f"  car owners (n {o['n']}), full controls: occasional {100*o['occasional'][0]:.1f} pp, regular {100*o['regular'][0]:.1f} pp "
          f"[{100*(o['regular'][0]-1.96*o['regular'][1]):.1f}, {100*(o['regular'][0]+1.96*o['regular'][1]):.1f}], joint p {o['p']:.4f}")

# %% [markdown]
# ## 8. Weekly car-commute days and commute days (Table 7 analogue)

# %%
cd = {}
for y, s in YEARS.items():
    for lab, f in [("demographic", BASE3), ("full", FULL3)]:
        for wt in (False, True):
            for out in ("cardays", "days"):
                r = smf.glm(f"{out} ~ " + f, s, family=sm.families.Poisson(), var_weights=s.w if wt else None).fit(cov_type="HC1")
                cd[f"{y}|{lab}|{'weighted' if wt else 'unweighted'}|{out}"] = dict(
                    occasional=contrast(r, s, "occasional", "none", wt), regular=contrast(r, s, "regular", "none", wt), p=wald(r, "tg"))
    g = s.groupby("tg").agg(days=("days", "mean"), car=("car", "mean"), cardays=("cardays", "mean"))
    cd[f"{y}|raw"] = g.to_dict()
    x, lo = g.loc["regular"], g.loc["none"]
    cd[f"{y}|decomposition regular vs none"] = dict(observed=x.cardays - lo.cardays, mode=(x.car - lo.car) * lo.days,
                                                    frequency=(x.days - lo.days) * lo.car, overlap=(x.car - lo.car) * (x.days - lo.days))
RESULTS["cardays"] = cd
print("\nDifferences in days per week from no telework [95% CI] (Poisson QML, HC1)")
for k, v in cd.items():
    if "raw" in k or "decomposition" in k:
        continue
    fm = lambda e: f"{e[0]: .2f} [{e[0]-1.96*e[1]: .2f}, {e[0]+1.96*e[1]: .2f}]"
    print(f"  {k:40s} occasional {fm(v['occasional'])}  regular {fm(v['regular'])}")
for y in YEARS:
    print(f"  {y} observed means by group:", {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in cd[f'{y}|raw'].items()})
    print(f"  {y} decomposition (regular vs none):", {k: round(v, 2) for k, v in cd[f'{y}|decomposition regular vs none'].items()})

# %% [markdown]
# ## 9. Reported change in household vehicles over the previous two years
#
# Asked of households that currently own a vehicle; households that gave up all vehicles are not observed.

# %%
vc = {}
for y, s in YEARS.items():
    v = s[s.vehicle_change.isin([1, 2, 3])].copy()
    v["red"] = (v.vehicle_change == 1).astype(int)
    noneed = "reduced_veh_3"
    v["red_noneed"] = ((v.vehicle_change == 1) & (v[noneed] == 1)).astype(int) if noneed in v else 0
    v["age4"] = pd.cut(v.agec, [0, 5, 6, 7, 11], labels=["18-34", "35-44", "45-54", "55+"])
    X = patsy.dmatrix("C(tg, Treatment('none')) + C(age4) + inc + edu + C(hboro)", v, return_type="dataframe")
    idx = [i for i, c in enumerate(X.columns) if "tg" in c]
    vc[y] = {"n": len(v), "n_by": v.tg.value_counts().reindex(["none", "occasional", "regular"]).tolist()}
    for out in ("red", "red_noneed"):
        vc[y][out] = dict(rates=v.groupby("tg")[out].mean().reindex(["none", "occasional", "regular"]).tolist(),
                          events=int(v[out].sum()), firth_p=firth_lr(v[out].values.astype(float), X.values, idx))
RESULTS["vehicle_change"] = vc
for y, v in vc.items():
    print(f"\n{y} vehicle change (n {v['n']}; none / occasional / regular {v['n_by']})")
    for out in ("red", "red_noneed"):
        r = v[out]; print(f"  {out:10s} rates {[round(x, 3) for x in r['rates']]}  events {r['events']}  Firth p {r['firth_p']:.3f}")

# %%
def conv(o):
    if isinstance(o, dict):
        return {str(k): conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [conv(x) for x in o]
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return conv(o.tolist())
    if isinstance(o, float) and np.isnan(o):
        return None
    return o


with open(os.path.join(RESULTSDIR, "robustness_2024.json"), "w") as fh:
    json.dump(conv(RESULTS), fh, indent=1, ensure_ascii=False)
print("\nwrote", os.path.join(RESULTSDIR, "robustness_2024.json"))
