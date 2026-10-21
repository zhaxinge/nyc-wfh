#!/usr/bin/env python
# coding: utf-8

# %% [markdown]
# # Replication: Breaking the Daily Routine (manuscript v4)
#
# Reproduces every table and figure in *Breaking the Daily Routine: Habit Disruption, Vehicle
# Ownership, and the Impact of Work-from-Home on Sustainable Commuting* (v4):
#
# | Section | Manuscript location |
# |---|---|
# | 1 | Analytic sample (N = 715), Table 1.1, Table 1.2 |
# | 2 | Table 2, Figure 2, leave-one-out, bootstrap vertex, three-level checks (§4.1) |
# | 3 | Table 3 — mediation through household vehicles (§4.2) |
# | 4 | Table 4 (age groups), Table 5 (workplace groups), Figure 3 (§4.3–4.4) |
# | 5 | Probability-weighted sensitivity models (§4.5) |
# | 6 | Appendix Tables A.1–A.5 |
#
# **Input:** 2022 NYC Citywide Mobility Survey, Person file, saved as
# `data/raw/Citywide_Mobility_Survey_-_Person_HOUSEHOLD.csv` (see the README).
#
# Every reported value is compared with the manuscript by `check(...)`. The run continues past a
# mismatch so that all mismatches are listed together, then stops with an error at the end.

# %%
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.special import expit
from scipy.stats import chi2, f_oneway, norm, pearsonr, spearmanr
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")
pd.set_option("display.width", 160)

# Paths are relative to the repository root; override with WFH_CSV / WFH_FIGDIR / WFH_RESULTSDIR.
ROOT = os.path.abspath("..") if os.path.basename(os.getcwd()) == "notebooks" else os.getcwd()
CSV = os.environ.get("WFH_CSV", os.path.join(ROOT, "data", "raw", "Citywide_Mobility_Survey_-_Person_HOUSEHOLD.csv"))
FIGDIR = os.environ.get("WFH_FIGDIR", os.path.join(ROOT, "figures"))
RESULTSDIR = os.environ.get("WFH_RESULTSDIR", os.path.join(ROOT, "results"))
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(RESULTSDIR, exist_ok=True)
B_BOOT = int(os.environ.get("WFH_BOOT", 1000))   # bootstrap resamples (manuscript: 1,000)

RESULTS = {}
MISMATCHES = []


def check(label, got, want, tol=0.001):
    """Compare a computed value with the manuscript value; record mismatches."""
    ok = abs(got - want) <= tol
    print(f"CHECK {label:60s} got {got: .4f}  manuscript {want: .4f}  {'OK' if ok else 'MISMATCH'}")
    if not ok:
        MISMATCHES.append((label, got, want))


def check_below(label, got, bound=0.001):
    """For values the manuscript reports as '< bound' (e.g. p < 0.001)."""
    ok = got < bound
    print(f"CHECK {label:60s} got {got: .4f}  manuscript < {bound}  {'OK' if ok else 'MISMATCH'}")
    if not ok:
        MISMATCHES.append((label, got, f"< {bound}"))


# %% [markdown]
# ## 1. Analytic sample and descriptive statistics (Tables 1.1 and 1.2)

# %%
df = pd.read_csv(CSV, low_memory=False)

KEY_VARS = ['wfh_policy', 'telework_freq', 'commute_freq', 'work_mode', 'education', 'age', 'gender',
            'income_broad', 'hh_cms_zone_gp', 'work_cms_zone', 'home_county', 'work_county']
for v in KEY_VARS:
    df.loc[df[v].isin([995, 999]), v] = np.nan      # 995 = missing, 999 = prefer not to answer

# Sustainable: walk, bus/shuttle, bicycle, rail/subway, micromobility, ferry (codebook work_mode codes)
df['sustainable_mode'] = df['work_mode'].isin([1, 102, 103, 105, 107, 108]).astype(float)
df['w'] = pd.to_numeric(df['person_weight'], errors='coerce')
df['education_clean'] = df['education']
df['income_clean'] = df['income_broad']
# Home and work counties share the FIPS classification, so cross-county is unambiguous
df['cross_county'] = np.where(df['home_county'].notna() & df['work_county'].notna(),
                              (df['home_county'] != df['work_county']).astype(float), np.nan)


def reverse_code(x):
    if x in [1, 2, 3, 4, 5, 6]:
        return 7 - x
    if x == 996:
        return 0
    return np.nan


df['tele_rev'] = df['telework_freq'].apply(reverse_code)       # actual telework frequency, 0-6
df['commute_rev'] = df['commute_freq'].apply(reverse_code)
df['wfh_binary'] = df['wfh_policy'].apply(lambda x: np.nan if pd.isna(x) else float(x > 1))
df['tele_binary'] = np.where(df['tele_rev'].isna(), np.nan, (df['tele_rev'] > 0).astype(float))

# WFHA: reverse-coded employer-required in-office days, 0 = 5+ days/week ... 6 = never required
df['wfh_policy_6'] = df['wfh_policy'].map({1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 996: 6})

chain = df.dropna(subset=['wfh_binary', 'tele_binary', 'commute_rev', 'work_mode', 'sustainable_mode', 'w'])
d = chain.dropna(subset=['sustainable_mode', 'wfh_policy_6', 'age', 'income_clean',
                         'cross_county', 'education_clean']).copy()

# Groups and markers
d['work_core'] = d['work_cms_zone'].isin([3, 4]).astype(float)    # Work_manhattan: CMS work zones 3-4
d['work_core4'] = (d['work_cms_zone'] == 4).astype(float)         # Manhattan core zone only
d['young'] = (d['age'] <= 6).astype(int)                          # 18-44 (age code <= 6)
d['young34'] = (d['age'] <= 5).astype(int)                        # 18-34, cut-point sensitivity
d['old'] = 1 - d['young']

# WFHA measurement variants
d['wfh_3level'] = pd.cut(d['wfh_policy_6'], [-1, 1, 3, 6], labels=False).astype(int)  # low 0-1, med 2-3, high 4-6
d['monthly_cadence'] = (d['wfh_policy_6'] >= 4).astype(float)

# Mediator and added controls
d['veh'] = d['num_vehicles'].clip(upper=2)                        # household vehicles, capped at 2+
d['man'] = np.where(d['gender'].isna(), np.nan, (d['gender'] == 2).astype(float))
d['kids'] = (d['num_kids'] > 0).astype(float)
d['parttime'] = (d['employment'] != 1).astype(float)
d['home_cty'] = d['home_county'].astype(int).astype(str)          # home borough
d['ind'] = d['industry'].astype(int)                              # 17 industry categories
d['w_norm'] = d['w'] / d['w'].mean()

RESULTS['N'] = len(d)
print(f"N = {len(d)}, weighted N = {d['w'].sum():,.0f}")
check("analytic sample N", len(d), 715, 0)
check("weighted N (millions)", d['w'].sum() / 1e6, 1.23, 0.005)
check("younger workers n", d['young'].sum(), 480, 0)
check("Manhattan workers n", d['work_core'].sum(), 514, 0)

# Table 1.1
t11 = d[['age', 'education_clean', 'income_clean', 'wfh_policy_6', 'cross_county', 'veh',
         'work_core', 'sustainable_mode']].agg(['mean', 'std', 'min', 'max']).T
print("\nTable 1.1")
print(t11.round(2).to_string())
for v, m, s in [('age', 6.08, 1.31), ('education_clean', 6.04, 1.28), ('income_clean', 4.55, 1.69),
                ('wfh_policy_6', 2.33, 1.44), ('cross_county', 0.55, 0.50), ('veh', 0.58, 0.68),
                ('work_core', 0.72, 0.45), ('sustainable_mode', 0.87, 0.34)]:
    check(f"Table 1.1 mean {v}", t11.loc[v, 'mean'], m, 0.006)
    check(f"Table 1.1 SD {v}", t11.loc[v, 'std'], s, 0.006)

# Table 1.2
CORR_VARS = ['sustainable_mode', 'wfh_policy_6', 'age', 'income_clean', 'education_clean',
             'cross_county', 'work_core', 'veh']
corr = d[CORR_VARS].corr()
pvals = pd.DataFrame([[pearsonr(d[a], d[b])[1] for b in CORR_VARS] for a in CORR_VARS],
                     index=CORR_VARS, columns=CORR_VARS)
stars = pvals.map(lambda p: '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else '')
print("\nTable 1.2")
print((corr.round(2).astype(str) + stars).where(np.tril(np.ones(corr.shape, bool)), '').to_string())
for (a, b), r in {('wfh_policy_6', 'sustainable_mode'): 0.11, ('cross_county', 'sustainable_mode'): 0.09,
                  ('age', 'sustainable_mode'): -0.12, ('work_core', 'sustainable_mode'): 0.41,
                  ('veh', 'sustainable_mode'): -0.35, ('veh', 'wfh_policy_6'): -0.16,
                  ('income_clean', 'education_clean'): 0.33, ('work_core', 'income_clean'): 0.25}.items():
    check(f"Table 1.2 r({a}, {b})", corr.loc[a, b], r, 0.006)

share_by_loc = d.groupby('work_core')['sustainable_mode'].mean()
check("sustainable share, Manhattan", share_by_loc[1.0], 0.955, 0.0006)
check("sustainable share, elsewhere", share_by_loc[0.0], 0.647, 0.0006)
check("Manhattan core zone n (of 514)", (d['work_cms_zone'] == 4).sum(), 470, 0)

# %% [markdown]
# ## Model helpers
#
# Logit models use HC3 standard errors. The WFHA average marginal effect combines the linear and
# squared terms, AME = mean[p(1 − p)(β₁ + 2β₂·WFHA)], with a delta-method standard error.

# %%
X = 'wfh_policy_6'
X2 = 'I(wfh_policy_6 ** 2)'
W = "wfh_policy_6 + I(wfh_policy_6**2)"
CTRL = "age + income_clean + education_clean + cross_county"     # pooled controls (Table 2)
GCTRL = "income_clean + education_clean + cross_county"          # group models: no age, no vehicles


def logit(formula, data):
    return smf.logit(formula, data=data).fit(disp=0, cov_type="HC3")


def wald(m, terms):
    """Joint Wald test that all `terms` are zero (uses the model's HC3 covariance)."""
    R = np.zeros((len(terms), len(m.params)))
    for i, t in enumerate(terms):
        R[i, m.params.index.get_loc(t)] = 1
    res = m.wald_test(R, scalar=True)
    return float(np.squeeze(res.statistic)), float(np.squeeze(res.pvalue))


def lr(full, reduced, data):
    F = smf.logit(full, data=data).fit(disp=0)
    R = smf.logit(reduced, data=data).fit(disp=0)
    stat = 2 * (F.llf - R.llf)
    k = int(F.df_model - R.df_model)
    return stat, k, chi2.sf(stat, k)


def _delta(m, fn):
    """Point estimate and delta-method SE of a scalar function of the parameters."""
    b = m.params.values
    est = fn(b)
    eps = 1e-6
    grad = np.array([(fn(b + eps * e) - fn(b - eps * e)) / (2 * eps) for e in np.eye(len(b))])
    se = float(np.sqrt(grad @ m.cov_params().values @ grad))
    return float(est), se, float(2 * norm.sf(abs(est / se)))


def ame_wfha(m):
    exog = m.model.exog
    names = list(m.params.index)
    i1, i2 = names.index(X), names.index(X2)
    w = exog[:, i1]

    def fn(b):
        p = expit(exog @ b)
        return np.mean(p * (1 - p) * (b[i1] + 2 * b[i2] * w))
    return _delta(m, fn)


def ame_discrete(m, var):
    """Average change in predicted probability when `var` moves from 0 to 1."""
    j = list(m.params.index).index(var)
    e1, e0 = m.model.exog.copy(), m.model.exog.copy()
    e1[:, j], e0[:, j] = 1, 0
    return _delta(m, lambda b: np.mean(expit(e1 @ b) - expit(e0 @ b)))


def vertex(m):
    return _delta(m, lambda b: -b[list(m.params.index).index(X)] / (2 * b[list(m.params.index).index(X2)]))


def coef_table(models, terms):
    rows = {}
    for t in terms:
        rows[t] = []
        for m in models:
            if t not in m.params:
                rows[t].append("—")
                continue
            p = m.pvalues[t]
            s = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''
            rows[t].append(f"{m.params[t]:.3f}{s} ({m.bse[t]:.3f})")
    return pd.DataFrame(rows).T


# %% [markdown]
# ## 2. Main association (H1): Table 2, Figure 2, and the §4.1 checks

# %%
m_lin = logit(f"sustainable_mode ~ {X} + {CTRL}", d)
m_quad = logit(f"sustainable_mode ~ {W} + {CTRL}", d)
print("Table 2")
print(coef_table([m_lin, m_quad], [X, X2, 'age', 'income_clean', 'education_clean', 'cross_county',
                                   'Intercept']).set_axis(['(1) Linear', '(2) Quadratic'], axis=1))
print(f"pseudo R2: {m_lin.prsquared:.3f} / {m_quad.prsquared:.3f}")

check("Table 2 linear: WFHA b", m_lin.params[X], 0.237)
check("Table 2 linear: WFHA SE", m_lin.bse[X], 0.109)
check("Table 2 linear: pseudo R2", m_lin.prsquared, 0.049)
check("Table 2 quadratic: WFHA b", m_quad.params[X], 0.837)
check("Table 2 quadratic: WFHA SE", m_quad.bse[X], 0.233)
check("Table 2 quadratic: WFHA^2 b", m_quad.params[X2], -0.105)
check("Table 2 quadratic: WFHA^2 SE", m_quad.bse[X2], 0.036)
check("Table 2 quadratic: WFHA^2 p", m_quad.pvalues[X2], 0.004)
check("Table 2 quadratic: age b", m_quad.params['age'], -0.261)
check("Table 2 quadratic: cross-county b", m_quad.params['cross_county'], 0.394)
check("Table 2 quadratic: constant", m_quad.params['Intercept'], 1.548)
check("Table 2 quadratic: pseudo R2", m_quad.prsquared, 0.062)

lr_q = lr(f"sustainable_mode ~ {W} + {CTRL}", f"sustainable_mode ~ {X} + {CTRL}", d)
check("LR test of quadratic term, chi2(1)", lr_q[0], 7.14, 0.006)
check("LR test of quadratic term, p", lr_q[2], 0.008)

v_est, v_se, _ = vertex(m_quad)
RESULTS['table2'] = dict(b1=m_quad.params[X], b2=m_quad.params[X2], vertex=v_est, vertex_se=v_se,
                         lr_quad=lr_q[0], lr_quad_p=lr_q[2])
check("Table 2 vertex", v_est, 3.97, 0.006)

# Bootstrap of the vertex and of the quadratic term's significance (1,000 resamples)
np.random.seed(11)
tps, n_sig = [], 0
for _ in range(B_BOOT):
    bs = d.sample(len(d), replace=True)
    try:
        mb = smf.logit(f"sustainable_mode ~ {W} + {CTRL}", data=bs).fit(disp=0)
    except Exception:
        continue
    n_sig += mb.pvalues[X2] < 0.05
    tps.append(-mb.params[X] / (2 * mb.params[X2]))
tp_lo, tp_hi = np.nanpercentile(tps, [2.5, 97.5])
RESULTS['vertex_boot_ci'] = [tp_lo, tp_hi]
RESULTS['boot_quad_sig_pct'] = 100 * n_sig / B_BOOT
print(f"vertex bootstrap 95% CI [{tp_lo:.2f}, {tp_hi:.2f}]; quadratic significant in {100 * n_sig / B_BOOT:.1f}%")
check("vertex bootstrap CI, lower", tp_lo, 3.22, 0.006)
check("vertex bootstrap CI, upper", tp_hi, 7.02, 0.006)
check("bootstrap: % resamples with significant quadratic", 100 * n_sig / B_BOOT, 74.3, 0.06)

# Leave-one-category-out
loo = {}
for lv in sorted(d[X].unique()):
    loo[int(lv)] = logit(f"sustainable_mode ~ {W} + {CTRL}", d[d[X] != lv]).pvalues[X2]
print("leave-one-category-out quadratic p:", {k: round(float(v), 3) for k, v in loo.items()})
RESULTS['loo_quad_p'] = loo
check("LOO: drop WFHA = 0, quadratic p", loo[0], 0.098)
check("LOO: drop WFHA = 5, quadratic p", loo[5], 0.014)
for lv in (1, 2, 3, 4, 5, 6):
    check(f"LOO: drop WFHA = {lv}, quadratic still significant", float(loo[lv] < 0.05), 1, 0)

# Three-level version
shares3 = d.groupby('wfh_3level')['sustainable_mode'].mean()
anova3 = f_oneway(*[d.loc[d['wfh_3level'] == k, 'sustainable_mode'] for k in (0, 1, 2)])
print(pairwise_tukeyhsd(d['sustainable_mode'], d['wfh_3level']).summary())
for k, s in zip((0, 1, 2), (0.765, 0.895, 0.897)):
    check(f"three-level share, level {k}", shares3[k], s, 0.0006)
check("three-level ANOVA F", anova3.statistic, 9.00, 0.006)
check_below("three-level ANOVA p", anova3.pvalue)

CAT = "C(wfh_3level, Treatment(reference=1))"
LOW, HIGH = f"{CAT}[T.0]", f"{CAT}[T.2]"
m_cat_loc = logit(f"sustainable_mode ~ {CAT} + {CTRL} + work_core", d)
check("location-adjusted categorical: low vs medium b", m_cat_loc.params[LOW], -0.556)
check("location-adjusted categorical: low vs medium p", m_cat_loc.pvalues[LOW], 0.040)
check("location-adjusted categorical: high vs medium p", m_cat_loc.pvalues[HIGH], 0.912)
w_cat = wald(m_cat_loc, [LOW, HIGH])
check("location-adjusted categorical: joint chi2(2)", w_cat[0], 4.82, 0.006)
check("location-adjusted categorical: joint p", w_cat[1], 0.090)


def predict_band(m, data, xcol=X):
    """Predicted probability over the WFHA range at mean covariate values, with 95% CI."""
    grid = pd.DataFrame({xcol: np.linspace(0, 6, 61)})
    for c in m.model.exog_names:
        if c not in ('Intercept', X, X2) and c in data:
            grid[c] = data[c].mean()
    sf = m.get_prediction(grid).summary_frame()
    return grid[xcol], sf['predicted'], sf['ci_lower'], sf['ci_upper']


def plot_curve(ax, m, data, label, color, solid=True):
    x, p, lo, hi = predict_band(m, data)
    ax.plot(x, p, color=color, lw=2, ls='-' if solid else '--', label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=0.15, lw=0)
    obs = data.groupby(X)['sustainable_mode'].agg(['mean', 'size'])
    ax.scatter(obs.index, obs['mean'], s=obs['size'] * 0.6, color=color, alpha=0.6, edgecolor='white', zorder=3)


fig, ax = plt.subplots(figsize=(6.5, 4.2))
plot_curve(ax, m_quad, d, 'All workers', '#1f5fa8')
ax.set_xlabel('WFH Frequency Allowance (WFHA, 0 = in office 5+ days/week … 6 = never required)')
ax.set_ylabel('P(sustainable commute)')
ax.set_ylim(0.4, 1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'fig2_pooled_wfha.png'), dpi=300)
plt.close(fig)
print("saved fig2_pooled_wfha.png")

# %% [markdown]
# ## 3. Mediation through household vehicles (Table 3)
#
# Path a: OLS of vehicles on WFHA (HC3). Total AME: logit without vehicles. Direct AME: logit with
# vehicles. Indirect AME = total − direct, with percentile bootstrap intervals (1,000 resamples).

# %%
GROUPS = {
    'All workers': (d, CTRL),
    'Younger (18-44)': (d[d['young'] == 1], GCTRL),
    'Older (45+)': (d[d['young'] == 0], GCTRL),
    'Manhattan': (d[d['work_core'] == 1], GCTRL),
    'Other locations': (d[d['work_core'] == 0], GCTRL),
}


def mediation(data, ctrl, extra="", boot=True, seed=7):
    a = smf.ols(f"veh ~ {X} + {ctrl}{extra}", data=data).fit(cov_type="HC3")
    tot = ame_wfha(logit(f"sustainable_mode ~ {W} + {ctrl}{extra}", data))
    dirc = ame_wfha(logit(f"sustainable_mode ~ {W} + {ctrl}{extra} + veh", data))
    out = dict(n=len(data), a=a.params[X], a_p=a.pvalues[X], total=tot[0], total_p=tot[2],
               direct=dirc[0], direct_p=dirc[2], indirect=tot[0] - dirc[0], share=(tot[0] - dirc[0]) / tot[0])
    if boot:
        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(B_BOOT):
            bs = data.sample(len(data), replace=True, random_state=int(rng.integers(1e9)))
            try:
                t = ame_wfha(smf.logit(f"sustainable_mode ~ {W} + {ctrl}{extra}", data=bs).fit(disp=0))[0]
                dr = ame_wfha(smf.logit(f"sustainable_mode ~ {W} + {ctrl}{extra} + veh", data=bs).fit(disp=0))[0]
                draws.append(t - dr)
            except Exception:
                continue
        out['ci'] = list(np.percentile(draws, [2.5, 97.5]))
    return out


med = {g: mediation(data, ctrl) for g, (data, ctrl) in GROUPS.items()}
RESULTS['table3'] = med
print(pd.DataFrame(med).T.round(3).to_string())

T3 = {  # n, path a, a p, total, direct, indirect, CI low, CI high, share
    'All workers': (715, -0.077, None, 0.044, 0.025, 0.019, 0.009, 0.030, 0.44),
    'Younger (18-44)': (480, -0.069, None, 0.054, 0.039, 0.014, 0.002, 0.030, 0.27),
    'Older (45+)': (235, -0.091, 0.007, 0.022, 0.000, 0.022, 0.004, 0.044, None),
    'Manhattan': (514, -0.049, 0.004, 0.000, -0.004, 0.003, -0.001, 0.010, None),
    'Other locations': (201, -0.142, None, 0.074, 0.039, 0.034, 0.010, 0.059, 0.47),
}
for g, (n, a, ap, tot, dr, ind, lo, hi, sh) in T3.items():
    r = med[g]
    check(f"Table 3 {g}: n", r['n'], n, 0)
    check(f"Table 3 {g}: path a", r['a'], a)
    if ap is None:
        check_below(f"Table 3 {g}: path a p", r['a_p'])
    else:
        check(f"Table 3 {g}: path a p", r['a_p'], ap)
    check(f"Table 3 {g}: total AME", r['total'], tot)
    check(f"Table 3 {g}: direct AME", r['direct'], dr)
    check(f"Table 3 {g}: indirect AME", r['indirect'], ind)
    # Percentile bounds depend on the random stream; the manuscript's came from a different one,
    # so they are checked to within Monte Carlo error rather than to the third decimal.
    check(f"Table 3 {g}: indirect CI lower", r['ci'][0], lo, 0.004)
    check(f"Table 3 {g}: indirect CI upper", r['ci'][1], hi, 0.004)
    if sh is not None:
        check(f"Table 3 {g}: share mediated", r['share'], sh, 0.006)
for g in ('Older (45+)', 'Manhattan'):
    check(f"Table 3 {g}: total AME not significant", float(med[g]['total_p'] >= 0.05), 1, 0)

med_boro = mediation(d, CTRL, " + C(home_cty)", boot=False)
med_core = mediation(d, CTRL, " + work_core", boot=False)
check("path a with home borough", med_boro['a'], -0.063)
check("path a with Manhattan workplace", med_core['a'], -0.065)
check("share mediated with home borough", med_boro['share'], 0.35, 0.006)

# Mechanical link between vehicles and mode (§4.2, Table A.5)
by_veh = d.groupby('veh')['sustainable_mode'].mean()
nonsust = d[d['sustainable_mode'] == 0]
check("sustainable share, 0 vehicles", by_veh[0], 0.976, 0.0006)
check("sustainable share, 1 vehicle", by_veh[1], 0.773, 0.0006)
check("sustainable share, 2+ vehicles", by_veh[2], 0.658, 0.0006)
check("non-sustainable commuters", len(nonsust), 94, 0)
check("non-sustainable commuters with zero vehicles", (nonsust['veh'] == 0).sum(), 9, 0)

# %% [markdown]
# ## 4. Group models: Table 4 (age), Table 5 (workplace), Figure 3

# %%
group_models = {g: logit(f"sustainable_mode ~ {W} + {GCTRL}", data)
                for g, (data, _) in GROUPS.items() if g != 'All workers'}
TERMS = [X, X2, 'income_clean', 'education_clean', 'cross_county', 'Intercept']
for title, pair in [("Table 4", ['Younger (18-44)', 'Older (45+)']), ("Table 5", ['Manhattan', 'Other locations'])]:
    print(title)
    tab = coef_table([group_models[g] for g in pair], TERMS).set_axis(pair, axis=1)
    tab.loc['Pseudo R2'] = [f"{group_models[g].prsquared:.3f}" for g in pair]
    tab.loc['Joint test of WFHA terms (p)'] = [f"{wald(group_models[g], [X, X2])[1]:.3f}" for g in pair]
    tab.loc['Vertex (SE)'] = ["{:.2f} ({:.2f})".format(*vertex(group_models[g])[:2]) for g in pair]
    print(tab.to_string(), "\n")

T45 = {  # b1, se1, b2, se2, joint p, vertex, vertex SE, pseudo R2, b1 p, b2 p
    'Younger (18-44)': (0.854, 0.297, -0.086, 0.047, 0.001, 4.99, 1.30, 0.065, 0.004, 0.069),
    'Older (45+)': (0.728, 0.409, -0.125, 0.066, 0.169, 2.91, 0.53, 0.046, 0.075, 0.059),
    'Manhattan': (0.175, 0.647, -0.037, 0.091, 0.790, 2.39, 3.32, 0.037, 0.787, 0.686),
    'Other locations': (0.506, 0.296, -0.044, 0.050, 0.039, 5.80, 3.72, 0.037, 0.088, 0.382),
}
RESULTS['tables4_5'] = {}
for g, (b1, s1, b2, s2, jp, vx, vse, r2, p1, p2) in T45.items():
    m = group_models[g]
    ve, vs, _ = vertex(m)
    jp_got = wald(m, [X, X2])[1]
    RESULTS['tables4_5'][g] = dict(b1=m.params[X], b2=m.params[X2], joint_p=jp_got, vertex=ve, vertex_se=vs)
    check(f"{g}: WFHA b", m.params[X], b1)
    check(f"{g}: WFHA SE", m.bse[X], s1)
    check(f"{g}: WFHA p", m.pvalues[X], p1)
    check(f"{g}: WFHA^2 b", m.params[X2], b2)
    check(f"{g}: WFHA^2 SE", m.bse[X2], s2)
    check(f"{g}: WFHA^2 p", m.pvalues[X2], p2)
    check(f"{g}: joint Wald p", jp_got, jp)
    check(f"{g}: vertex", ve, vx, 0.006)
    check(f"{g}: vertex SE", vs, vse, 0.006)
    check(f"{g}: pseudo R2", m.prsquared, r2)
check("Table 4 younger: constant", group_models['Younger (18-44)'].params['Intercept'], -0.395)
check("Table 5 Manhattan: education b", group_models['Manhattan'].params['education_clean'], 0.393)
check("older workers at WFHA = 5", (d[(d['young'] == 0)][X] == 5).sum(), 8, 0)
check("older workers at WFHA = 6", (d[(d['young'] == 0)][X] == 6).sum(), 8, 0)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, (panel, pair) in zip(axes, [("(a) Age", ['Younger (18-44)', 'Older (45+)']),
                                    ("(b) Workplace location", ['Manhattan', 'Other locations'])]):
    for g, color in zip(pair, ['#1f5fa8', '#c1440e']):
        sig = wald(group_models[g], [X, X2])[1] < 0.05
        plot_curve(ax, group_models[g], GROUPS[g][0], g, color, solid=sig)
    ax.set_title(panel, loc='left')
    ax.set_xlabel('WFHA (0–6)')
    ax.legend(fontsize=8, loc='lower right')
axes[0].set_ylabel('P(sustainable commute)')
axes[0].set_ylim(0.2, 1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'fig3_group_wfha.png'), dpi=300)
plt.close(fig)
print("saved fig3_group_wfha.png")

# Adding Work_manhattan to the pooled model (§4.4)
m_quad_loc = logit(f"sustainable_mode ~ {W} + {CTRL} + work_core", d)
check("pooled + Manhattan: WFHA b", m_quad_loc.params[X], 0.535)
check("pooled + Manhattan: WFHA^2 b", m_quad_loc.params[X2], -0.067)
check("pooled + Manhattan: reduction in WFHA b", 1 - m_quad_loc.params[X] / m_quad.params[X], 0.36, 0.006)

# %% [markdown]
# ## 5. Probability-weighted sensitivity models (§4.5)
#
# Person weights normalized to mean one, entered as `var_weights` with HC3 covariance.

# %%
m_w_loc = smf.glm(f"sustainable_mode ~ {W} + {CTRL} + work_core", data=d, family=sm.families.Binomial(),
                  var_weights=d['w_norm']).fit(cov_type='HC3')
w_w = wald(m_w_loc, [X, X2])
m_w_age = smf.glm(f"sustainable_mode ~ {X}*age + income_clean + education_clean + cross_county + work_core",
                  data=d, family=sm.families.Binomial(), var_weights=d['w_norm']).fit(cov_type='HC3')
RESULTS['weighted'] = dict(b1=m_w_loc.params[X], b2=m_w_loc.params[X2], core=m_w_loc.params['work_core'],
                           joint=w_w, wfh_x_age=m_w_age.params[f'{X}:age'], wfh_x_age_p=m_w_age.pvalues[f'{X}:age'])
check("weighted: WFHA b", m_w_loc.params[X], 0.499)
check("weighted: WFHA SE", m_w_loc.bse[X], 0.468)
check("weighted: WFHA p", m_w_loc.pvalues[X], 0.286)
check("weighted: WFHA^2 b", m_w_loc.params[X2], -0.079)
check("weighted: WFHA^2 SE", m_w_loc.bse[X2], 0.072)
check("weighted: WFHA^2 p", m_w_loc.pvalues[X2], 0.275)
check("weighted: joint Wald chi2(2)", w_w[0], 1.20, 0.006)
check("weighted: joint Wald p", w_w[1], 0.549)
check("weighted: Work_manhattan b", m_w_loc.params['work_core'], 1.766)
check("weighted: Work_manhattan SE", m_w_loc.bse['work_core'], 0.444)
check("weighted: WFHA x age b", m_w_age.params[f'{X}:age'], -0.187)
check("weighted: WFHA x age SE", m_w_age.bse[f'{X}:age'], 0.089)
check("weighted: WFHA x age p", m_w_age.pvalues[f'{X}:age'], 0.035)

# %% [markdown]
# ## 6. Appendix tables
#
# ### Table A.1 — joint significance of the WFH terms under alternative measurements and controls
#
# Pooled column: controls of Table 2 (with age). Group columns: no age term.

# %%
MEASURES = {
    'quad': (W, [X, X2]),
    'tele': ("tele_rev + I(tele_rev**2)", ['tele_rev', 'I(tele_rev ** 2)']),
    '3lvl': (CAT, [LOW, HIGH]),
    'monthly': ("monthly_cadence", ['monthly_cadence']),
}
COLS = {
    'Pooled': ('all', None),
    'Age 18-44': ('young', 1), 'Age 45+': ('young', 0),
    'Manhattan': ('work_core', 1), 'Other': ('work_core', 0),
}


def joint_p(measure, data, ctrl):
    rhs, terms = MEASURES[measure]
    return wald(logit(f"sustainable_mode ~ {rhs} + {ctrl}", data), terms)[1]


def a1_row(measure="quad", extra="", data=d, cols=COLS):
    row = {}
    for col, (var, val) in cols.items():
        sub = data if var == 'all' else data[data[var] == val]
        ctrl = (CTRL if var == 'all' else GCTRL) + extra
        row[col] = joint_p(measure, sub, ctrl)
    return row


d_gkp = d.dropna(subset=['man'])
A1 = {
    'Primary: WFHA, quadratic': a1_row(),
    'Actual telework frequency, quadratic': a1_row('tele'),
    'WFHA, three levels': a1_row('3lvl'),
    'WFHA, monthly or less vs. weekly': a1_row('monthly'),
    'Age split 34/35': a1_row(cols={'Age 18-44': ('young34', 1), 'Age 45+': ('young34', 0)}),
    'Manhattan core zone only': a1_row(cols={'Manhattan': ('work_core4', 1), 'Other': ('work_core4', 0)}),
    '+ home borough': a1_row(extra=" + C(home_cty)"),
    '+ home borough, gender, children, part-time': a1_row(extra=" + C(home_cty) + man + kids + parttime",
                                                          data=d_gkp),
    '+ home borough, industry': a1_row(extra=" + C(home_cty) + C(ind)"),
    '+ household vehicles': a1_row(extra=" + veh"),
}
A1 = pd.DataFrame(A1).T[list(COLS)]
RESULTS['tableA1'] = A1.to_dict()
print("Table A.1")
print(A1.round(3).to_string())

A1_MS = {
    'Primary: WFHA, quadratic': (0.001, 0.001, 0.169, 0.790, 0.039),
    'Actual telework frequency, quadratic': (0.006, 0.005, 0.070, 0.021, 0.052),
    'WFHA, three levels': (0.003, '<', 0.722, 0.633, 0.018),
    'WFHA, monthly or less vs. weekly': (0.379, 0.132, 0.569, 0.726, 0.184),
    'Age split 34/35': (None, 0.017, 0.004, None, None),
    'Manhattan core zone only': (None, None, None, 0.727, 0.053),
    '+ home borough': (0.009, 0.021, 0.125, 0.771, 0.080),
    '+ home borough, gender, children, part-time': (0.009, 0.040, 0.170, 0.537, 0.053),
    '+ home borough, industry': (0.050, 0.155, 0.089, 0.239, 0.056),
    '+ household vehicles': (0.028, 0.019, 0.176, 0.451, 0.298),
}
for spec, wants in A1_MS.items():
    for col, want in zip(COLS, wants):
        if want is None:
            continue
        if want == '<':
            check_below(f"A.1 {spec} | {col}", A1.loc[spec, col])
        else:
            check(f"A.1 {spec} | {col}", A1.loc[spec, col], want)
check("A.1 gender/children/part-time sample n", len(d_gkp), 694, 0)
for col, n in zip(['Age 18-44', 'Age 45+', 'Manhattan', 'Other'], [462, 232, 499, 195]):
    var, val = COLS[col]
    check(f"A.1 gender/children/part-time n, {col}", (d_gkp[var] == val).sum(), n, 0)

# %% [markdown]
# ### Table A.2 — pooled tests of whether the WFH association differs across groups
#
# Likelihood-ratio tests in pooled unweighted logit models. The age test enters age once, as the
# group (no continuous age control); the Manhattan test uses the controls of Table 2. The income
# test interacts the WFHA terms with a high/low median split of income, adding Work_manhattan.

# %%
A2 = {}
for nm, g, ctrl in [('age group', 'C(old)', GCTRL), ('Manhattan workplace', 'C(work_core)', CTRL)]:
    base = f"sustainable_mode ~ {W} + {g} + {ctrl}"
    A2[f'WFHA terms x {nm}'] = lr(f"sustainable_mode ~ ({W})*{g} + {ctrl}", base, d)
    A2[f'WFHA linear term x {nm}'] = lr(f"sustainable_mode ~ {W} + {X}:{g} + {g} + {ctrl}", base, d)
d['inc_high'] = (d['income_clean'] > d['income_clean'].median()).astype(int)
INC_CTRL = "age + education_clean + cross_county + work_core"
A2['WFHA terms x income (median split)'] = lr(f"sustainable_mode ~ ({W})*C(inc_high) + {INC_CTRL}",
                                              f"sustainable_mode ~ {W} + C(inc_high) + {INC_CTRL}", d)
A2['Quadratic term added, without Manhattan'] = lr_q
A2['Quadratic term added, with Manhattan'] = lr(f"sustainable_mode ~ {W} + {CTRL} + work_core",
                                                f"sustainable_mode ~ {X} + {CTRL} + work_core", d)
RESULTS['tableA2'] = A2
print("Table A.2")
for k, (s, df_, p) in A2.items():
    print(f"  {k:45s} chi2({df_}) = {s:5.2f}  p = {p:.3f}")
for k, (s, p) in {'WFHA terms x age group': (6.28, 0.043), 'WFHA linear term x age group': (6.00, 0.014),
                  'WFHA terms x Manhattan workplace': (0.91, 0.634),
                  'WFHA linear term x Manhattan workplace': (0.77, 0.380),
                  'WFHA terms x income (median split)': (3.16, 0.206),
                  'Quadratic term added, with Manhattan': (2.51, 0.113)}.items():
    check(f"A.2 {k}: chi2", A2[k][0], s, 0.006)
    check(f"A.2 {k}: p", A2[k][2], p)

# %% [markdown]
# ### Table A.3 — group models that include household vehicles

# %%
veh_models = {'All workers': logit(f"sustainable_mode ~ {W} + {CTRL} + veh", d)}
veh_models.update({g: logit(f"sustainable_mode ~ {W} + {GCTRL} + veh", data)
                   for g, (data, _) in GROUPS.items() if g != 'All workers'})
tabA3 = coef_table(list(veh_models.values()), [X, X2, 'veh']).set_axis(list(veh_models), axis=1)
tabA3.loc['Pseudo R2'] = [f"{m.prsquared:.3f}" for m in veh_models.values()]
tabA3.loc['Joint test of WFHA terms (p)'] = [f"{wald(m, [X, X2])[1]:.3f}" for m in veh_models.values()]
print("Table A.3")
print(tabA3.to_string())
A3_MS = {  # WFHA b, SE; WFHA^2 b, SE; vehicles b, SE; pseudo R2
    'All workers': (0.696, 0.260, -0.103, 0.040, -1.578, 0.179, 0.204),
    'Younger (18-44)': (0.862, 0.334, -0.104, 0.051, -1.683, 0.235, 0.220),
    'Older (45+)': (0.488, 0.434, -0.111, 0.071, -1.679, 0.296, 0.212),
    'Manhattan': (0.112, 0.666, -0.041, 0.094, -1.437, 0.337, 0.127),
    'Other locations': (0.514, 0.330, -0.077, 0.053, -1.520, 0.278, 0.175),
}
for g, (b1, s1, b2, s2, bv, sv, r2) in A3_MS.items():
    m = veh_models[g]
    for lab, got, want in [('WFHA b', m.params[X], b1), ('WFHA SE', m.bse[X], s1),
                           ('WFHA^2 b', m.params[X2], b2), ('WFHA^2 SE', m.bse[X2], s2),
                           ('vehicles b', m.params['veh'], bv), ('vehicles SE', m.bse['veh'], sv),
                           ('pseudo R2', m.prsquared, r2)]:
        check(f"A.3 {g}: {lab}", got, want)

# %% [markdown]
# ### Table A.4 — average marginal effects of WFHA and Manhattan workplace under added controls

# %%
A4_SPECS = [
    ('Demographic controls only', CTRL, d),
    ('+ vehicles', CTRL + " + veh", d),
    ('+ home borough', CTRL + " + veh + C(home_cty)", d),
    ('+ gender, children, part-time', CTRL + " + veh + C(home_cty) + man + kids + parttime", d_gkp),
    ('+ industry', CTRL + " + veh + C(home_cty) + C(ind)", d),
]
rows = []
for nm, ctrl, data in A4_SPECS:
    no = ame_wfha(logit(f"sustainable_mode ~ {W} + {ctrl}", data))
    m_loc = logit(f"sustainable_mode ~ {W} + {ctrl} + work_core", data)
    lo = ame_wfha(m_loc)
    core = ame_discrete(m_loc, 'work_core')
    rows.append(dict(spec=nm, n=len(data), ame_no_loc=no[0], se_no_loc=no[1], ame_loc=lo[0], se_loc=lo[1],
                     attenuation=1 - lo[0] / no[0], ame_core=core[0], se_core=core[1]))
A4 = pd.DataFrame(rows).set_index('spec')
RESULTS['tableA4'] = A4.to_dict()
print("Table A.4")
print(A4.round(3).to_string())
A4_MS = {  # n, WFHA AME no loc (SE), WFHA AME + Manhattan (SE), attenuation, Manhattan AME (SE)
    'Demographic controls only': (715, 0.044, 0.013, 0.024, 0.011, 0.46, 0.298, 0.040),
    '+ vehicles': (715, 0.025, 0.012, 0.012, 0.010, 0.51, 0.221, 0.036),
    '+ home borough': (715, 0.020, 0.011, 0.010, 0.010, 0.50, 0.226, 0.038),
    '+ gender, children, part-time': (694, 0.021, 0.011, 0.011, 0.010, 0.48, 0.229, 0.039),
    '+ industry': (715, 0.011, 0.011, 0.005, 0.011, 0.59, 0.192, 0.035),
}
for nm, (n, a0, s0, a1, s1, att, ac, sc) in A4_MS.items():
    r = A4.loc[nm]
    check(f"A.4 {nm}: n", r['n'], n, 0)
    check(f"A.4 {nm}: WFHA AME, no location", r['ame_no_loc'], a0)
    check(f"A.4 {nm}: WFHA AME SE, no location", r['se_no_loc'], s0)
    check(f"A.4 {nm}: WFHA AME, + Manhattan", r['ame_loc'], a1)
    check(f"A.4 {nm}: WFHA AME SE, + Manhattan", r['se_loc'], s1)
    check(f"A.4 {nm}: attenuation", r['attenuation'], att, 0.006)
    check(f"A.4 {nm}: Manhattan AME", r['ame_core'], ac)
    check(f"A.4 {nm}: Manhattan AME SE", r['se_core'], sc)

# %% [markdown]
# ### Bootstrap AME contrasts (§4.3, §4.4–4.5, §5.1; Table A.4 note)
#
# Percentile intervals from 1,000 resamples. Bounds depend on the random stream, so they are checked
# to within Monte Carlo error (±0.004, or ±0.006 for values reported to two decimals; ±0.02 for the
# attenuation share, reported in whole percent).

# %%
def boot(fn, data, seed=7):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B_BOOT):
        try:
            out.append(fn(data.sample(len(data), replace=True, random_state=int(rng.integers(1e9)))))
        except Exception:
            continue
    return np.array(out)


def pctl(x):
    return list(np.percentile(x, [2.5, 97.5]))


# (a) Pooled WFHA AME once Manhattan is added, and the attenuation share
def loc_attenuation(data):
    a0 = ame_wfha(smf.logit(f"sustainable_mode ~ {W} + {CTRL}", data=data).fit(disp=0))[0]
    a1 = ame_wfha(smf.logit(f"sustainable_mode ~ {W} + {CTRL} + work_core", data=data).fit(disp=0))[0]
    return a1, 1 - a1 / a0


bs_att = boot(loc_attenuation, d)
ci_ame_loc, ci_att = pctl(bs_att[:, 0]), pctl(bs_att[:, 1])
print(f"WFHA AME + Manhattan CI [{ci_ame_loc[0]:.3f}, {ci_ame_loc[1]:.3f}]; "
      f"attenuation CI [{ci_att[0]:.0%}, {ci_att[1]:.0%}]")
check("WFHA AME + Manhattan, CI lower", ci_ame_loc[0], 0.003, 0.004)
check("WFHA AME + Manhattan, CI upper", ci_ame_loc[1], 0.047, 0.004)
check("attenuation, CI lower", ci_att[0], 0.27, 0.02)
check("attenuation, CI upper", ci_att[1], 0.89, 0.02)

# (b) Age-group AMEs: stratified models with the Table 4 controls plus Work_manhattan
F_AGE = f"sustainable_mode ~ {W} + {GCTRL} + work_core"
young, older = d[d['young'] == 1], d[d['young'] == 0]
ame_y = ame_wfha(logit(F_AGE, young))[0]
ame_o = ame_wfha(logit(F_AGE, older))[0]
# Resample the pooled sample, then split by age, so the difference has a joint distribution
bs_age = boot(lambda b: [ame_wfha(smf.logit(F_AGE, data=b[b['young'] == v]).fit(disp=0))[0] for v in (1, 0)], d)
bs_y, bs_o = bs_age[:, 0], bs_age[:, 1]
ci_diff = pctl(bs_y - bs_o)
print(f"WFHA AME younger {ame_y:.3f} {np.round(pctl(bs_y), 3)}, older {ame_o:.3f} {np.round(pctl(bs_o), 3)}, "
      f"difference {ame_y - ame_o:.3f} {np.round(ci_diff, 3)}")
check("AME younger", ame_y, 0.034)
check("AME younger, CI lower", pctl(bs_y)[0], 0.010, 0.004)
check("AME younger, CI upper", pctl(bs_y)[1], 0.059, 0.004)
check("AME older", ame_o, 0.002)
check("AME older, CI lower", pctl(bs_o)[0], -0.042, 0.004)
check("AME older, CI upper", pctl(bs_o)[1], 0.045, 0.004)
check("AME difference, younger - older", ame_y - ame_o, 0.033)
check("AME difference, CI lower", ci_diff[0], -0.016, 0.004)
check("AME difference, CI upper", ci_diff[1], 0.082, 0.004)
check("AME difference not significant", float(ci_diff[0] < 0 < ci_diff[1]), 1, 0)

# (c) Manhattan versus income: one model with income in three groups (bands 1-3, 4-6, 7-8)
INC3 = ['low', 'mid', 'high']
d['inc3'] = pd.Categorical(pd.cut(d['income_clean'], [0, 3, 6, 99], labels=INC3), categories=INC3)
F_INC3 = f"sustainable_mode ~ {W} + age + C(inc3) + education_clean + cross_county + work_core"


def loc_vs_income(data):
    m = smf.logit(F_INC3, data=data).fit(disp=0)
    on, off, hi, lo = (data.copy() for _ in range(4))
    on['work_core'], off['work_core'] = 1.0, 0.0
    hi['inc3'] = pd.Categorical(['high'] * len(data), categories=INC3)
    lo['inc3'] = pd.Categorical(['low'] * len(data), categories=INC3)
    return (m.predict(on) - m.predict(off)).mean(), (m.predict(hi) - m.predict(lo)).mean()


core_ame, inc_ame = loc_vs_income(d)
bs_li = boot(loc_vs_income, d)
ci_core, ci_inc = pctl(bs_li[:, 0]), pctl(bs_li[:, 1])
print(f"Manhattan AME {core_ame:.3f} {np.round(ci_core, 3)}; high vs low income AME {inc_ame:.3f} {np.round(ci_inc, 3)}")
check("Manhattan AME (three income groups)", core_ame, 0.29, 0.006)
check("Manhattan AME, CI lower", ci_core[0], 0.21, 0.006)
check("Manhattan AME, CI upper", ci_core[1], 0.37, 0.006)
check("high vs low income AME", inc_ame, -0.02, 0.006)
check("high vs low income AME, CI lower", ci_inc[0], -0.10, 0.004)
check("high vs low income AME, CI upper", ci_inc[1], 0.06, 0.006)

RESULTS['bootstrap_contrasts'] = dict(
    ame_loc_ci=ci_ame_loc, attenuation_ci=ci_att,
    ame_younger=ame_y, ame_younger_ci=pctl(bs_y), ame_older=ame_o, ame_older_ci=pctl(bs_o),
    ame_age_diff=ame_y - ame_o, ame_age_diff_ci=ci_diff,
    manhattan_ame=core_ame, manhattan_ame_ci=ci_core, income_high_low_ame=inc_ame, income_high_low_ci=ci_inc)

# %% [markdown]
# ### Table A.5 — diagnostics for the mediator (household vehicles)

# %%
check("A.5 Pearson r(vehicles, sustainable)", pearsonr(d['veh'], d['sustainable_mode'])[0], -0.35, 0.006)
check("A.5 Spearman r(vehicles, sustainable)", spearmanr(d['veh'], d['sustainable_mode'])[0], -0.35, 0.006)
check("A.5 r(vehicles, WFHA)", d['veh'].corr(d[X]), -0.16, 0.006)
check("A.5 r(WFHA, sustainable)", d[X].corr(d['sustainable_mode']), 0.11, 0.006)
r2_veh = smf.logit("sustainable_mode ~ veh", data=d).fit(disp=0).prsquared
r2_wfh = smf.logit(f"sustainable_mode ~ {X}", data=d).fit(disp=0).prsquared
r2_both = smf.logit(f"sustainable_mode ~ {X} + veh", data=d).fit(disp=0).prsquared
print(f"pseudo R2: vehicles only {r2_veh:.3f} / WFHA only {r2_wfh:.3f} / both {r2_both:.3f}")
check("A.5 pseudo R2, vehicles only", r2_veh, 0.144)
check("A.5 pseudo R2, WFHA only", r2_wfh, 0.017)
check("A.5 pseudo R2, both", r2_both, 0.149)
vif_X = sm.add_constant(d[[X, 'age', 'income_clean', 'education_clean', 'cross_county', 'veh']])
vifs = [variance_inflation_factor(vif_X.values, i) for i in range(1, vif_X.shape[1])]
print(f"VIF range {min(vifs):.2f}-{max(vifs):.2f}")
check("A.5 VIF minimum", min(vifs), 1.04, 0.006)
check("A.5 VIF maximum", max(vifs), 1.20, 0.006)

# %% [markdown]
# ## Save results and report

# %%
with open(os.path.join(RESULTSDIR, 'results_v4.json'), 'w') as f:
    json.dump(RESULTS, f, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
print("saved results_v4.json")

if MISMATCHES:
    print(f"\n{len(MISMATCHES)} value(s) do not reproduce the manuscript:")
    for label, got, want in MISMATCHES:
        print(f"  {label}: got {got}, manuscript {want}")
    raise AssertionError(f"{len(MISMATCHES)} manuscript value(s) did not reproduce")
print("\nAll manuscript values reproduce.")
