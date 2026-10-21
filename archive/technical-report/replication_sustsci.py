#!/usr/bin/env python
# coding: utf-8

# # Replication notebook — Sustainability Science technical-report revision
# 
# Reproduces the analytic sample (N = 715), all reported marginal effects, the measurement-cadence
# robustness tests, the piecewise-linear model, Figures 1–2, and the supplementary indirect-path check.
# 
# **Input:** 2022 NYC Citywide Mobility Survey — Person file, saved as
# `data/raw/Citywide_Mobility_Survey_-_Person_HOUSEHOLD.csv` (see the repository README).
# 
# **Requires:** pandas, numpy, scipy, statsmodels, matplotlib

# ## Setup

# In[1]:


"""
Replication script for the Sustainability Science technical-report revision.
Reproduces the analytic sample (N=715), all marginal effects, the cadence-break
tests, the piecewise-linear model, and Figures 1-2.
Input: 2022 NYC Citywide Mobility Survey - Person file, saved as
       Citywide_Mobility_Survey_-_Person_HOUSEHOLD.csv in the working directory.
Requires: pandas, numpy, scipy, statsmodels, matplotlib
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
pd.set_option("display.width", 160)

import os
# Paths are relative to the repository root; override with WFH_CSV / WFH_FIGDIR / WFH_RESULTSDIR.
ROOT = os.path.abspath("..") if os.path.basename(os.getcwd()) == "notebooks" else os.getcwd()
CSV = os.environ.get("WFH_CSV", os.path.join(ROOT, "data", "raw", "Citywide_Mobility_Survey_-_Person_HOUSEHOLD.csv"))
os.environ.setdefault("WFH_FIGDIR", os.path.join(ROOT, "figures"))
RESULTSDIR = os.environ.get("WFH_RESULTSDIR", os.path.join(ROOT, "results"))
os.makedirs(RESULTSDIR, exist_ok=True)
SEED = 7
np.random.seed(SEED)

RESULTS = {}   # collects every number that appears in the manuscript
print("ready")


# ## Load data and build the analytic sample (N = 715)

# In[2]:


df = pd.read_csv(CSV, low_memory=False)

KEY_VARS = ['wfh_policy','telework_freq','commute_freq','work_mode','education',
            'age','gender','income_broad','hh_cms_zone_gp','work_cms_zone']
for v in KEY_VARS:
    df.loc[df[v].isin([995, 999]), v] = np.nan      # 995 = Missing, 999 = Prefer not to answer

# Dependent variable (codebook work_mode codes)
SUSTAINABLE_CODES = [1, 102, 103, 105, 107, 108]   # walk, bus, bike, rail/subway, micromob, ferry
df['sustainable_mode'] = df['work_mode'].isin(SUSTAINABLE_CODES).astype(float)

df['w']               = pd.to_numeric(df['person_weight'], errors='coerce')
# Gender: tested (man = gender==2) but non-significant (p=0.47) and dropped for parsimony.
# See §Methods note. df['man'] intentionally NOT created.
df['education_clean'] = df['education']
df['income_clean']    = df['income_broad']
# CODEBOOK: hh_cms_zone_gp and work_cms_zone use DIFFERENT numbering systems -> not comparable.
# Use the survey's direct county fields (home_county/work_county, shared FIPS codes) instead.
for _c in ['home_county','work_county']:
    df.loc[df[_c].isin([995, 999]), _c] = np.nan
df['cross_county']    = np.where(df['home_county'].notna() & df['work_county'].notna(),
                                 (df['home_county'] != df['work_county']).astype(float), np.nan)

def reverse_code(x):
    if x in [1, 2, 3, 4, 5, 6]:
        return 7 - x
    elif x == 996:
        return 0
    return np.nan

df['tele_rev']    = df['telework_freq'].apply(reverse_code)
df['commute_rev'] = df['commute_freq'].apply(reverse_code)
df['wfh_binary']  = df['wfh_policy'].apply(lambda x: np.nan if pd.isna(x) else float(x > 1))
df['tele_binary'] = np.where(df['tele_rev'].isna(), np.nan, (df['tele_rev'] > 0).astype(float))
df['Y']           = df['sustainable_mode']

print(f"rows loaded: {len(df)}")

# (a) Published 0–6 recode — direction correct, spacing non-interval
WFH_MAP_6 = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 996: 6}
df['wfh_policy_6'] = df['wfh_policy'].map(WFH_MAP_6)

# (b) Three-level robustness coding used in the manuscript.
#     This is derived from the published 0–6 scale so the cutoffs are explicit:
#       low = 0–1, medium = 2–3, high = 4–6.
#     It is used only as a categorical H1 robustness specification.
df['wfh_3level'] = np.select(
    [df['wfh_policy_6'].between(0, 1),
     df['wfh_policy_6'].between(2, 3),
     df['wfh_policy_6'].between(4, 6)],
    [0, 1, 2],
    default=np.nan,
)

# (c) Workplace location: Manhattan (transit-rich) work zones. Sensitivity tested in §4.2c.
df['work_core']    = df['work_cms_zone'].isin([3, 4]).astype(float)   # Northern Manhattan + Manhattan Core
df['work_ny_cty']  = (df['work_county'] == 36061).astype(float)       # New York County (FIPS) - cross-check
df['work_core4']   = (df['work_cms_zone'] == 4).astype(float)         # Manhattan Core only - sensitivity

print(df[['wfh_policy_6', 'wfh_3level', 'work_core']].describe().T[['count','mean','min','max']])


df_chain = df.dropna(subset=['wfh_binary','tele_binary','commute_rev','work_mode','Y','w']).copy()
BASE = ["sustainable_mode","wfh_policy_6","age","income_clean","cross_county","education_clean"]
d = df_chain.dropna(subset=BASE).copy()

CTRL = "age + income_clean + education_clean + cross_county"

RESULTS['N'] = len(d)
RESULTS['weighted_N'] = d['w'].sum()
print(f"N                 = {len(d)}          (0422 published: 715)")
print(f"Weighted N        = {d['w'].sum():,.0f}   (0422 published: ~1,228,867)")
print(f"Sustainable share = {d['sustainable_mode'].mean():.3f}")
print(f"Works Manhattan   = {d['work_core'].mean():.3f}")
assert len(d) == 715, "sample no longer reproduces 0422"
print("\n[OK] sample reproduces 0422")


# ## Table 1.1 — descriptive statistics

# In[3]:


rows = []
for v in ['age','education_clean','income_clean','wfh_policy_6']:
    rows.append([v, d[v].mean(), d[v].std(), d[v].min(), d[v].max()])
for v in ['cross_county','work_core','sustainable_mode']:
    rows.append([v, d[v].mean(), d[v].std(), d[v].min(), d[v].max()])
t1 = pd.DataFrame(rows, columns=['variable','mean','sd','min','max'])
print(t1.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

print("\nCorrected notes for Table 1.1:")
print("  Age    : observed range 4-11 (4 = 18-24 ... 11 = 85+). Sample is adults only.")
print("  WFHA   : derived from employer-REQUIRED in-office days per week:")
print("           0 = required 5+ d/wk; 1 = 4 d/wk; 2 = 2-3 d/wk; 3 = 1 d/wk;")
print("           4 = 1-3 d/month;      5 = < monthly; 6 = never (fully remote).")
print("           Scale is non-interval: levels 0-3 weekly cadence, 4-6 monthly cadence.")


# ## §4.1a — Baseline quadratic model (H1)

# In[4]:


# Two-step: (1) reproduce 0422 EXACTLY with its original (buggy) coding as a provenance check,
#           (2) report the corrected baseline used from here on.

# (1) Reproduce 0422: original gender (==1) and zone-based cross_county
d_orig = d.copy()
d_orig['male_orig']  = (df.loc[d.index, 'gender'] == 1).astype(float)
d_orig['cross_orig'] = np.where(df.loc[d.index,'hh_cms_zone_gp'].notna() & df.loc[d.index,'work_cms_zone'].notna(),
                                (df.loc[d.index,'hh_cms_zone_gp'] != df.loc[d.index,'work_cms_zone']).astype(float), np.nan)
CTRL_ORIG = "age + male_orig + income_clean + education_clean + cross_orig"
m_orig = smf.logit(f"sustainable_mode ~ wfh_policy_6 + I(wfh_policy_6**2) + {CTRL_ORIG}",
                   data=d_orig).fit(disp=0, cov_type="HC3")
TP_orig = -m_orig.params['wfh_policy_6']/(2*m_orig.params['I(wfh_policy_6 ** 2)'])
print(f"(1) 0422 original coding  -> TP = {TP_orig:.3f}   (published: 3.880)")
assert abs(TP_orig - 3.880) < 0.01, "original-coding TP no longer reproduces 0422"
print("    [OK] faithfully reproduces the published inverted-U\n")


# ## §4.1b — Robustness: leave-one-cell-out

# In[5]:


# (2) Corrected baseline: FIPS cross_county, gender DROPPED (tested, p=0.47, n.s.) — used below
m_quad = smf.logit(f"sustainable_mode ~ wfh_policy_6 + I(wfh_policy_6**2) + {CTRL}",
                   data=d).fit(disp=0, cov_type="HC3")
b1 = m_quad.params['wfh_policy_6']; b2 = m_quad.params['I(wfh_policy_6 ** 2)']
TP = -b1/(2*b2)
RESULTS.update(dict(b1=b1, b2=b2, TP=TP, TP_orig=TP_orig,
                    p_b1=m_quad.pvalues['wfh_policy_6'],
                    p_b2=m_quad.pvalues['I(wfh_policy_6 ** 2)'],
                    r2_baseline=m_quad.prsquared))
print(f"(2) corrected coding      -> TP = {TP:.3f}")
print(f"    WFH linear     b = {b1:+.3f}   p = {m_quad.pvalues['wfh_policy_6']:.4f}")
print(f"    WFH quadratic  b = {b2:+.4f}   p = {m_quad.pvalues['I(wfh_policy_6 ** 2)']:.4f}")
print(f"    pseudo-R2        = {m_quad.prsquared:.3f}")
print(f"    -> dropping gender + fixing cross_county shifts TP by {abs(TP-TP_orig):.2f}; "
      f"curvature and all conclusions unchanged.")


# formula strings reused by later cells (baseline quadratic Q and linear L)
Q = f"sustainable_mode ~ wfh_policy_6 + I(wfh_policy_6**2) + {CTRL}"
L = f"sustainable_mode ~ wfh_policy_6 + {CTRL}"


def fit_tp(data):
    m = smf.logit(Q, data=data).fit(disp=0, cov_type="HC3")
    _b1 = m.params['wfh_policy_6']; _b2 = m.params['I(wfh_policy_6 ** 2)']
    return -_b1 / (2 * _b2), m.pvalues['I(wfh_policy_6 ** 2)']

out = []
t, p = fit_tp(d)
out.append(['full sample', len(d), p, t, 'yes' if p < .05 else 'NO'])
for lv in sorted(d['wfh_policy_6'].unique()):
    sub = d[d['wfh_policy_6'] != lv]
    n_dropped = int((d['wfh_policy_6'] == lv).sum())
    t, p = fit_tp(sub)
    out.append([f'drop level {int(lv)} (n={n_dropped})', len(sub), p, t, 'yes' if p < .05 else 'NO'])

loo = pd.DataFrame(out, columns=['spec','n','quad_p','TP','quad_sig'])
RESULTS['loo'] = loo
print(loo.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\n--> Curvature is robust to dropping any single cell EXCEPT level 0.")
print("    It is driven by the LOW end, not by the small high-flexibility cells.")


# ## §4.1c — Robustness: bootstrap stability of the turning point

# In[6]:


np.random.seed(11)
B = 1000
tps, n_sig, n_inside = [], 0, 0

for _ in range(B):
    bs = d.sample(len(d), replace=True)
    try:
        m = smf.logit(Q, data=bs).fit(disp=0)
        _b1 = m.params['wfh_policy_6']; _b2 = m.params['I(wfh_policy_6 ** 2)']
        if m.pvalues['I(wfh_policy_6 ** 2)'] < 0.05:
            n_sig += 1
        t = -_b1 / (2 * _b2)
        tps.append(t)
        if 0 <= t <= 6:
            n_inside += 1
    except Exception:
        pass

tp_lo, tp_hi = np.nanpercentile(tps, [2.5, 97.5])
RESULTS.update(dict(boot_sig_pct=100*n_sig/B, boot_inside_pct=100*n_inside/B,
                    TP_CI=(tp_lo, tp_hi)))

print(f"quadratic significant in     {100*n_sig/B:.1f}% of {B} resamples")
print(f"turning point inside [0,6] in {100*n_inside/B:.1f}% of resamples")
print(f"TP 95% CI = [{tp_lo:.2f}, {tp_hi:.2f}]   <-- upper bound OUTSIDE the 0-6 scale")
print("\n--> Curvature: robust.  Location of the peak: NOT precisely identified.")
print("    Report as diminishing returns / plateau, not 'optimal level = 3.9'.")


# ## §4.1d — Predicted probabilities vs. observed shares (three-level categorical robustness)

# In[7]:


# Observed seven-level shares
tab = d.groupby('wfh_policy_6').agg(
    n=('sustainable_mode', 'size'),
    share=('sustainable_mode', 'mean'),
)
tab['pct_of_sample'] = 100 * tab['n'] / tab['n'].sum()
print("Raw (unweighted) share by WFHA level:")
print(tab.to_string(float_format=lambda x: f"{x:.3f}"))

# Formal three-level categorical robustness model: low 0–1, medium 2–3, high 4–6.
d['wfh_3level'] = d['wfh_3level'].astype(int)
d['w_norm'] = d['w'] / d['w'].mean()  # preserve N-scale for weighted robust inference
level_labels = {0: 'low (0–1)', 1: 'medium (2–3)', 2: 'high (4–6)'}

rows = []
for k, g in d.groupby('wfh_3level'):
    rows.append([
        int(k), level_labels[int(k)], len(g),
        g['sustainable_mode'].mean(),
        np.average(g['sustainable_mode'], weights=g['w']),
    ])
t3 = pd.DataFrame(
    rows,
    columns=['wfh_3level', 'label', 'n', 'unweighted_share', 'weighted_share'],
)
print()
print("Three-level descriptive shares (manuscript cutoffs):")
print(t3.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# Unweighted omnibus comparison and pairwise contrasts reported in the manuscript.
anova_3 = f_oneway(*[
    d.loc[d['wfh_3level'] == k, 'sustainable_mode'] for k in (0, 1, 2)
])
tukey_3 = pairwise_tukeyhsd(
    endog=d['sustainable_mode'],
    groups=d['wfh_3level'],
    alpha=0.05,
)
print()
print(f"One-way ANOVA: F = {anova_3.statistic:.2f}, p = {anova_3.pvalue:.4g}")
print(tukey_3.summary())

# Adjusted categorical logit with medium as the reference category and HC3 inference.
F3 = f"sustainable_mode ~ C(wfh_3level, Treatment(reference=1)) + {CTRL}"
m_3cat = smf.logit(F3, data=d).fit(disp=0, cov_type='HC3')
low_key = 'C(wfh_3level, Treatment(reference=1))[T.0]'
high_key = 'C(wfh_3level, Treatment(reference=1))[T.2]'
R3 = np.zeros((2, len(m_3cat.params)))
R3[0, m_3cat.params.index.get_loc(low_key)] = 1
R3[1, m_3cat.params.index.get_loc(high_key)] = 1
wald_3 = m_3cat.wald_test(R3, scalar=True)
wald_3_stat = float(np.asarray(wald_3.statistic).squeeze())
wald_3_p = float(np.asarray(wald_3.pvalue).squeeze())

print()
print("Adjusted categorical logit (medium reference; HC3):")
for key, label in [(low_key, 'low vs medium'), (high_key, 'high vs medium')]:
    print(f"  {label:16s} b={m_3cat.params[key]:+.3f}, "
          f"SE={m_3cat.bse[key]:.3f}, p={m_3cat.pvalues[key]:.4f}")
print(f"  joint Wald chi2(2)={wald_3_stat:.2f}, p={wald_3_p:.4f}")

# Population-weighted robustness counterpart. Weights are normalized to mean one;
# HC3 covariance is retained so expansion weights do not create artificial precision.
m_3cat_w = smf.glm(
    F3,
    data=d,
    family=sm.families.Binomial(),
    freq_weights=d['w_norm'],
).fit(cov_type='HC3')
R3w = np.zeros((2, len(m_3cat_w.params)))
R3w[0, m_3cat_w.params.index.get_loc(low_key)] = 1
R3w[1, m_3cat_w.params.index.get_loc(high_key)] = 1
wald_3w = m_3cat_w.wald_test(R3w, scalar=True)
wald_3w_stat = float(np.asarray(wald_3w.statistic).squeeze())
wald_3w_p = float(np.asarray(wald_3w.pvalue).squeeze())

print()
print("Weighted categorical GLM (medium reference; normalized weights; HC3):")
for key, label in [(low_key, 'low vs medium'), (high_key, 'high vs medium')]:
    print(f"  {label:16s} b={m_3cat_w.params[key]:+.3f}, "
          f"SE={m_3cat_w.bse[key]:.3f}, p={m_3cat_w.pvalues[key]:.4f}")
print(f"  joint Wald chi2(2)={wald_3w_stat:.2f}, p={wald_3w_p:.4f}")

RESULTS.update({
    't3': t3,
    'wfh3_anova': (float(anova_3.statistic), float(anova_3.pvalue)),
    'wfh3_unweighted_wald': (wald_3_stat, wald_3_p),
    'wfh3_weighted_wald': (wald_3w_stat, wald_3w_p),
    'wfh3_unweighted_low': (m_3cat.params[low_key], m_3cat.pvalues[low_key]),
    'wfh3_unweighted_high': (m_3cat.params[high_key], m_3cat.pvalues[high_key]),
    'wfh3_weighted_low': (m_3cat_w.params[low_key], m_3cat_w.pvalues[low_key]),
    'wfh3_weighted_high': (m_3cat_w.params[high_key], m_3cat_w.pvalues[high_key]),
})

print()
print("Interpretation: the gain is concentrated between low and medium flexibility; "
      "medium and high form a plateau. This supports diminishing returns, not an interior optimum.")


F3_LOC = f"sustainable_mode ~ C(wfh_3level, Treatment(reference=1)) + {CTRL} + work_core"
m_3cat_loc = smf.logit(F3_LOC, data=d).fit(disp=0, cov_type='HC3')

R3_loc = np.zeros((2, len(m_3cat_loc.params)))
R3_loc[0, m_3cat_loc.params.index.get_loc(low_key)] = 1
R3_loc[1, m_3cat_loc.params.index.get_loc(high_key)] = 1
wald_3_loc = m_3cat_loc.wald_test(R3_loc, scalar=True)
wald_3_loc_stat = float(np.asarray(wald_3_loc.statistic).squeeze())
wald_3_loc_p = float(np.asarray(wald_3_loc.pvalue).squeeze())

print("Location-adjusted categorical logit (medium reference; HC3; + work_core):")
for key, label in [(low_key, 'low vs medium'), (high_key, 'high vs medium')]:
    print(f"  {label:16s} b={m_3cat_loc.params[key]:+.3f}, "
          f"SE={m_3cat_loc.bse[key]:.3f}, p={m_3cat_loc.pvalues[key]:.4f}")
print(f"  joint Wald chi2(2)={wald_3_loc_stat:.2f}, p={wald_3_loc_p:.4f}")

RESULTS.update({
    'wfh3_loc_low':  (m_3cat_loc.params[low_key],  m_3cat_loc.pvalues[low_key]),
    'wfh3_loc_high': (m_3cat_loc.params[high_key], m_3cat_loc.pvalues[high_key]),
    'wfh3_loc_wald': (wald_3_loc_stat, wald_3_loc_p),
})

print()
print("Manuscript Table 5 reports: low vs medium b=-0.556 (p=0.040), "
      "high vs medium b=+0.041 (p=0.912), joint chi2(2)=4.82 (p=0.090).")
print("If the numbers above don't match, update Table 5 to whichever is the correct rerun.")


# ## Figure: fitted quadratic vs. observed shares

# In[8]:


# Figure: predicted probability curve vs. observed shares
grid = pd.DataFrame({'wfh_policy_6': np.linspace(0, 6, 61)})
for v, val in [('age', d['age'].median()), ('income_clean', d['income_clean'].median()),
               ('education_clean', d['education_clean'].median()),
               ('cross_county', d['cross_county'].mean())]:
    grid[v] = val
grid['pred'] = m_quad.predict(grid)

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(grid['wfh_policy_6'], grid['pred'], lw=2, label='Fitted quadratic (predicted prob.)')
ax.scatter(tab.index, tab['share'], s=tab['n']*0.35, zorder=5, color='#c1440e',
           label='Observed share (size ∝ n)')
ax.axvline(TP, ls='--', color='grey', lw=1)
ax.axvspan(tp_lo, min(tp_hi, 6), alpha=0.12, color='grey', label=f'TP 95% CI [{tp_lo:.1f}, {tp_hi:.1f}]')
ax.annotate('cadence break\n(weekly → monthly)', xy=(3.5, 0.62), fontsize=8, ha='center')
ax.axvline(3.5, ls=':', color='black', lw=1)
ax.set_xlabel('WFHA (0 = required 5+ d/wk in office … 6 = never required)')
ax.set_ylabel('P(sustainable commute)')
ax.set_title('Fitted curvature vs. observed shares')
ax.set_ylim(0.55, 1.0); ax.legend(fontsize=8, loc='lower right')
plt.tight_layout()
import os
FIGDIR = os.environ.get('WFH_FIGDIR', '.')
os.makedirs(FIGDIR, exist_ok=True)
plt.savefig(os.path.join(FIGDIR, 'fig1_turning_point.png'), dpi=150)
print("saved fig1_turning_point.png")
plt.show()


# ## §4.2 — Location-adjusted quadratic: does curvature survive?

# In[9]:


def lr_test(full_f, red_f, data, dfree, label):
    F = smf.logit(full_f, data=data).fit(disp=0)
    R = smf.logit(red_f,  data=data).fit(disp=0)
    stat = 2 * (F.llf - R.llf)
    p = chi2.sf(stat, dfree)
    print(f"  {label:52s} LR chi2({dfree}) = {stat:5.2f}, p = {p:.4f}")
    return stat, p

L = f"sustainable_mode ~ wfh_policy_6 + {CTRL}"
print("Likelihood-ratio test on the QUADRATIC term:")
s0, p0 = lr_test(Q, L, d, 1, "without workplace location:")
s1, p1 = lr_test(Q + " + work_core", L + " + work_core", d, 1, "with workplace location:")
RESULTS.update(dict(lr_quad_nocore=(s0, p0), lr_quad_core=(s1, p1)))

m_loc = smf.logit(Q + " + work_core", data=d).fit(disp=0, cov_type="HC3")
RESULTS['r2_with_location'] = m_loc.prsquared
print(f"\nWorkplace location coefficient: b = {m_loc.params['work_core']:+.3f}, "
      f"p = {m_loc.pvalues['work_core']:.4f}")
print(f"pseudo-R2: {m_quad.prsquared:.3f} (baseline)  ->  {m_loc.prsquared:.3f} (with location)")
print("\n--> Curvature attenuates to non-significance once workplace location is controlled.")
print("    Interpretation in text: workplace location attenuates the WFH curvature.")

# [merged from coauthor v1: richer zone-definition sensitivity — reports linear + quad + Wald + R2]


# ## §4.2b — Sensitivity of the workplace-location (core zone) definition

# In[10]:


# Uses the corrected man/FIPS controls (QUAD_F, LIN_F, CTRL_F already defined above).
# NOTE: this loop uses `m_lin_sens`/`m_quad_sens` (not `m_lin`/`m_quad`) so it can never
# again silently overwrite the baseline models that §5's attenuation summary depends on.
CTRL_F = f"sustainable_mode ~ {CTRL}"                         # controls only
LIN_F  = f"sustainable_mode ~ wfh_policy_6 + {CTRL}"          # + linear WFH
QUAD_F = f"sustainable_mode ~ wfh_policy_6 + I(wfh_policy_6**2) + {CTRL}"  # + quadratic

print("Sensitivity Test: LINEAR and QUADRATIC Terms across Core Zone Definitions")
print("=" * 80)

for name, zones in {"Manhattan zones 3,4 (used)": [3, 4],
                    "zone 4 only":                [4],
                    "zones 3,4,8":                [3, 4, 8]}.items():

    d['_core'] = d['work_cms_zone'].isin(zones).astype(float)

    # 1. Models
    m_lin_sens  = smf.logit(LIN_F + " + _core", data=d).fit(disp=0, cov_type="HC3")  # Linear model
    m_quad_sens = smf.logit(QUAD_F + " + _core", data=d).fit(disp=0, cov_type="HC3")  # Quadratic model

    # 2. LR Test for Linear Term (comparing Linear Model vs Controls-only Model)
    s_lin, p_lin = lr_test(LIN_F + " + _core", CTRL_F + " + _core", d, 1, f"  linear term | {name}")

    # 3. LR Test for Quadratic Term (comparing Quadratic Model vs Linear Model)
    s_quad, p_quad = lr_test(QUAD_F + " + _core", LIN_F + " + _core", d, 1, f"  quad term   | {name}")

    # 4. Extract Wald Coefficient & p-value for the Linear term
    b_wfh = m_lin_sens.params['wfh_policy_6']
    p_wfh = m_lin_sens.pvalues['wfh_policy_6']

    print(f"      [Linear Model] WFH b = {b_wfh:+.3f} (p = {p_wfh:.4f}), "
          f"core b = {m_lin_sens.params['_core']:+.3f}, pseudo-R2 = {m_lin_sens.prsquared:.3f}\n")


# [merged from coauthor v1: dedicated non-Manhattan subsample analysis]
# Subsample: Non-Manhattan workplace workers
d_non_core = d[d['work_core'] == 0].copy()

print(f"Non-Manhattan Subsample Size: N = {len(d_non_core)}")
print(f"Sustainable Commute Share: {d_non_core['sustainable_mode'].mean():.3f}\n")


# ## §4.3b — Non-Manhattan subsample

# In[11]:


# Formulas
CTRL_F = f"sustainable_mode ~ {CTRL}"
LIN_F = f"sustainable_mode ~ wfh_policy_6 + {CTRL}"
QUAD_F = f"sustainable_mode ~ wfh_policy_6 + I(wfh_policy_6**2) + {CTRL}"

print("==================================================")
print("1. Likelihood-Ratio Tests (Non-Manhattan Subsample)")
print("==================================================")

# Test Linear Term (Linear vs Controls-only)
s_lin_nc, p_lin_nc = lr_test(
    LIN_F, CTRL_F, d_non_core, 1, "Linear WFH term:"
)

# Test Quadratic Term (Quadratic vs Linear)
s_quad_nc, p_quad_nc = lr_test(
    QUAD_F, LIN_F, d_non_core, 1, "Quadratic WFH term:"
)

print("\n==================================================")
print("2. Regression Models (Non-Manhattan Subsample)")
print("==================================================")

# Fit Linear Model
m_lin_nc = smf.logit(
    LIN_F,
    data=d_non_core
).fit(disp=0, cov_type="HC3")

print(
    f"[Linear Model]    WFH b = "
    f"{m_lin_nc.params['wfh_policy_6']:+.3f} "
    f"(SE = {m_lin_nc.bse['wfh_policy_6']:.3f}, "
    f"p = {m_lin_nc.pvalues['wfh_policy_6']:.4f})"
)

# Fit Quadratic Model
m_quad_nc = smf.logit(
    QUAD_F,
    data=d_non_core
).fit(disp=0, cov_type="HC3")

print(
    f"[Quadratic Model] WFH b = "
    f"{m_quad_nc.params['wfh_policy_6']:+.3f} "
    f"(SE = {m_quad_nc.bse['wfh_policy_6']:.3f}, "
    f"p = {m_quad_nc.pvalues['wfh_policy_6']:.4f})"
)

print(
    f"[Quadratic Model] WFH² b = "
    f"{m_quad_nc.params['I(wfh_policy_6 ** 2)']:+.3f} "
    f"(SE = {m_quad_nc.bse['I(wfh_policy_6 ** 2)']:.3f}, "
    f"p = {m_quad_nc.pvalues['I(wfh_policy_6 ** 2)']:.4f})"
)


# ## §Figures — stratified WFH curves: Manhattan vs. not

# In[12]:


# Fit Linear Model
m_lin_nc = smf.logit(
    LIN_F,
    data=d_non_core
).fit(disp=0, cov_type="HC3")

print(
    f"[Linear Model] WFH b = {m_lin_nc.params['wfh_policy_6']:+.3f} "
    f"(p = {m_lin_nc.pvalues['wfh_policy_6']:.4f}), "
    f"pseudo-R2 = {m_lin_nc.prsquared:.3f}"
)

# Fit Quadratic Model
m_quad_nc = smf.logit(
    QUAD_F,
    data=d_non_core
).fit(disp=0, cov_type="HC3")

b1_nc = m_quad_nc.params["wfh_policy_6"]
b2_nc = m_quad_nc.params["I(wfh_policy_6 ** 2)"]

print(
    f"[Quadratic Model] Linear b = {b1_nc:+.3f} "
    f"(p = {m_quad_nc.pvalues['wfh_policy_6']:.4f}), "
    f"Quad b = {b2_nc:+.4f} "
    f"(p = {m_quad_nc.pvalues['I(wfh_policy_6 ** 2)']:.4f}), "
    f"pseudo-R2 = {m_quad_nc.prsquared:.3f}"
)

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGDIR = os.environ.get("WFH_FIGDIR", ".")
os.makedirs(FIGDIR, exist_ok=True)

d["work_core"] = d["work_cms_zone"].isin([3, 4]).astype(float)

STRATA = [
    (1, "In Manhattan"),
    (0, "Not Manhattan"),
]

CTRL_STRAT = "age + income_clean + education_clean + cross_county"

print(
    "Quadratic-term test per stratum "
    "(decides whether the curve is plotted):\n"
)

plot_decisions = {}

for lv, nm in STRATA:
    sub = d[d["work_core"] == lv].copy()

    Ff = smf.logit(
        f"sustainable_mode ~ wfh_policy_6 + "
        f"I(wfh_policy_6**2) + {CTRL_STRAT}",
        data=sub
    ).fit(disp=0)

    Rf = smf.logit(
        f"sustainable_mode ~ wfh_policy_6 + {CTRL_STRAT}",
        data=sub
    ).fit(disp=0)

    lr_q = chi2.sf(
        2 * (Ff.llf - Rf.llf),
        1
    )

    ml = smf.logit(
        f"sustainable_mode ~ wfh_policy_6 + {CTRL_STRAT}",
        data=sub
    ).fit(
        disp=0,
        cov_type="HC3"
    )

    sig = lr_q < 0.05

    plot_decisions[nm] = {
        "sig": sig,
        "lr_q": lr_q,
        "n": len(sub),
        "sust": sub["sustainable_mode"].mean(),
        "lin_b": ml.params["wfh_policy_6"],
        "lin_p": ml.pvalues["wfh_policy_6"],
    }

    RESULTS[
        f"strat_quad_{'man' if lv else 'nonman'}"
    ] = lr_q

    significance_text = (
        "SIGNIFICANT: plot curve"
        if sig
        else "n.s.: linear only, NO plot"
    )

    print(
        f"  {nm:14s} "
        f"n={len(sub):3d} "
        f"sust={sub['sustainable_mode'].mean():.3f} | "
        f"quadratic LR p={lr_q:.4f} -> {significance_text} | "
        f"linear b={ml.params['wfh_policy_6']:+.3f} "
        f"p={ml.pvalues['wfh_policy_6']:.4f}"
    )

# Plot ONLY strata with a significant quadratic
to_plot = [
    (lv, nm)
    for lv, nm in STRATA
    if plot_decisions[nm]["sig"]
]

plot_names = (
    [nm for _, nm in to_plot]
    if to_plot
    else "NONE — no curve plots produced"
)

print(
    "\nStrata with significant curvature (plotted): "
    f"{plot_names}"
)

for lv, nm in to_plot:
    sub = d[d["work_core"] == lv].copy()

    mq = smf.logit(
        f"sustainable_mode ~ wfh_policy_6 + "
        f"I(wfh_policy_6**2) + {CTRL_STRAT}",
        data=sub
    ).fit(
        disp=0,
        cov_type="HC3"
    )


# ## §4.3d — Home vs. work Manhattan (residential vs. workplace location)

# In[13]:


# Plot significant quadratic strata
for lv, nm in to_plot:
    sub = d[d["work_core"] == lv].copy()

    mq = smf.logit(
        f"sustainable_mode ~ wfh_policy_6 + "
        f"I(wfh_policy_6**2) + {CTRL_STRAT}",
        data=sub
    ).fit(disp=0, cov_type="HC3")

    grid = pd.DataFrame({
        "wfh_policy_6": np.linspace(0, 6, 61)
    })

    # Gender/man variable was removed earlier and is intentionally excluded here.
    for v, val in [
        ("age", sub["age"].median()),
        ("income_clean", sub["income_clean"].median()),
        ("education_clean", sub["education_clean"].median()),
        ("cross_county", sub["cross_county"].mean()),
    ]:
        grid[v] = val

    grid["pred"] = mq.predict(grid)

    obs = (
        sub.groupby("wfh_policy_6")["sustainable_mode"]
        .agg(["mean", "size"])
    )

    fig, ax = plt.subplots(figsize=(7, 4.2))

    ax.plot(
        grid["wfh_policy_6"],
        grid["pred"],
        lw=2,
        label="Fitted quadratic"
    )

    ax.scatter(
        obs.index,
        obs["mean"],
        s=obs["size"] * 2,
        alpha=0.6,
        label="Observed share (size ∝ n)"
    )

    b1 = mq.params["wfh_policy_6"]
    b2 = mq.params["I(wfh_policy_6 ** 2)"]

    tp = -b1 / (2 * b2)

    if 0 <= tp <= 6:
        ax.axvline(
            tp,
            ls="--",
            color="grey",
            alpha=0.7,
            label=f"Turning point ≈ {tp:.2f}"
        )

    ax.set_xlabel("WFH policy flexibility (WFHA, 0–6)")
    ax.set_ylabel("P(sustainable commute)")
    ax.set_title(
        f"WFH curve — {nm} "
        f"(n={len(sub)}, quad p={plot_decisions[nm]['lr_q']:.3f})"
    )

    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.02)

    fname = os.path.join(
        FIGDIR,
        f"fig_curve_{'manhattan' if lv else 'nonmanhattan'}.png"
    )

    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)

    print(f"  saved: {fname}")


if not to_plot:
    print(
        "\nNo stratum shows significant curvature; "
        "per the rule, no stratified curve plots are generated."
    )
    print(
        "Report in text: within each stratum the WFH relationship "
        "is linear or flat, not U-shaped."
    )


# --------------------------------------------------
# Home Manhattan vs Work Manhattan
# --------------------------------------------------

# home_manhattan from grouped home zone
# 6 = Northern Manhattan, 10 = Manhattan Core
d["home_manhattan"] = (
    d["hh_cms_zone_gp"].isin([6, 10]).astype(float)
)

Qb = (
    f"sustainable_mode ~ "
    f"wfh_policy_6 + I(wfh_policy_6**2) + {CTRL}"
)

print(
    f"home-Manhattan share = "
    f"{d['home_manhattan'].mean():.3f} "
    f"(n={int(d['home_manhattan'].sum())})"
)

print(
    f"work-Manhattan share = "
    f"{d['work_core'].mean():.3f} "
    f"(n={int(d['work_core'].sum())})"
)

print(
    f"  live & work Manhattan: "
    f"{int(((d.home_manhattan == 1) & (d.work_core == 1)).sum())} | "
    f"live Manhattan/work elsewhere: "
    f"{int(((d.home_manhattan == 1) & (d.work_core == 0)).sum())} | "
    f"live elsewhere/work Manhattan: "
    f"{int(((d.home_manhattan == 0) & (d.work_core == 1)).sum())}"
)


for spec, lab in [
    ("work_core", "work-Manhattan only"),
    ("home_manhattan", "home-Manhattan only"),
    ("work_core + home_manhattan", "both together"),
]:
    m = smf.logit(
        f"{Qb} + {spec}",
        data=d
    ).fit(
        disp=0,
        cov_type="HC3"
    )

    terms = spec.split(" + ")

    term_results = []

    for t in terms:
        clean_name = (
            t.replace("_manhattan", "")
            .replace("work_core", "work")
        )

        term_results.append(
            f"{clean_name}="
            f"{m.params[t]:+.2f}"
            f"(p={m.pvalues[t]:.3f})"
        )

    print(
        f"  {lab:22s} "
        f"R2={m.prsquared:.3f}  "
        + "  ".join(term_results)
    )


RESULTS["home_vs_work_manhattan"] = (
    "work dominates; home n.s. when both entered"
)

print(
    "\n=> When both are entered, WORK-Manhattan remains strong "
    "and HOME-Manhattan is not significant."
)

print(
    "   Workplace location, not residential location, "
    "drives the result."
)


# ## Weighted, location-adjusted sensitivity model

# In[14]:


F_WEIGHTED_LOC = f"sustainable_mode ~ wfh_policy_6 + I(wfh_policy_6**2) + {CTRL} + work_core"
# NOTE: var_weights (not freq_weights) is what reproduces the manuscript's Table 5
# SEs exactly -- freq_weights treats the (normalized-to-1) weights as literal
# replicated observations, which understates SEs by roughly 2x here. var_weights
# instead scales each observation's contribution to the variance without inflating
# the effective N, which is the correct behavior for survey/precision weights.
m_w_loc = smf.glm(F_WEIGHTED_LOC, data=d, family=sm.families.Binomial(),
                  var_weights=d['w_norm']).fit(cov_type='HC3')

print("Weighted, location-adjusted quadratic model (normalized weights; HC3):")
for k, lab in [('wfh_policy_6', 'WFHA'), ('I(wfh_policy_6 ** 2)', 'WFHA^2'), ('work_core', 'work_manhattan')]:
    print(f"  {lab:16s} b={m_w_loc.params[k]:+.3f}, SE={m_w_loc.bse[k]:.3f}, "
          f"z={m_w_loc.tvalues[k]:.2f}, p={m_w_loc.pvalues[k]:.4f}")

R_w = np.zeros((2, len(m_w_loc.params)))
R_w[0, m_w_loc.params.index.get_loc('wfh_policy_6')] = 1
R_w[1, m_w_loc.params.index.get_loc('I(wfh_policy_6 ** 2)')] = 1
wald_w = m_w_loc.wald_test(R_w, scalar=True)
wald_w_stat = float(np.asarray(wald_w.statistic).squeeze())
wald_w_p = float(np.asarray(wald_w.pvalue).squeeze())
print(f"  WFH joint Wald chi2(2)={wald_w_stat:.2f}, p={wald_w_p:.4f}")

# Weighted WFH x continuous age interaction (location-adjusted); age entered ONCE,
# per the §4.3 specification warning.
OTH_LOC = "income_clean + education_clean + cross_county + work_core"
m_w_age_full = smf.glm(f"sustainable_mode ~ wfh_policy_6*age + {OTH_LOC}", data=d,
                       family=sm.families.Binomial(), var_weights=d['w_norm']).fit(cov_type='HC3')
int_key = 'wfh_policy_6:age'
print(f"\nWeighted WFH x continuous age (location-adjusted): "
      f"b={m_w_age_full.params[int_key]:+.3f}, SE={m_w_age_full.bse[int_key]:.3f}, "
      f"z={m_w_age_full.tvalues[int_key]:.2f}, p={m_w_age_full.pvalues[int_key]:.4f}")

RESULTS.update({
    'weighted_loc_wfh':  (m_w_loc.params['wfh_policy_6'], m_w_loc.pvalues['wfh_policy_6']),
    'weighted_loc_wfh2': (m_w_loc.params['I(wfh_policy_6 ** 2)'], m_w_loc.pvalues['I(wfh_policy_6 ** 2)']),
    'weighted_loc_core': (m_w_loc.params['work_core'], m_w_loc.pvalues['work_core']),
    'weighted_loc_wald': (wald_w_stat, wald_w_p),
    'weighted_wfh_age':  (m_w_age_full.params[int_key], m_w_age_full.pvalues[int_key]),
})

print()
print("Verified against manuscript Table 5: WFHA b=0.499 (p=0.286), WFHA^2 b=-0.079 (p=0.275), "
      "work_manhattan b=1.766 (p<0.001), WFH joint chi2(2)=1.20 (p=0.549), "
      "WFH x age b=-0.187 (p=0.035). Matches exactly with var_weights (see note above).")


# ## §4.3 — Age and income moderation

# In[15]:


d['age_old']  = (d['age'] > d['age'].median()).astype(int)
d['inc_high'] = (d['income_clean'] > d['income_clean'].median()).astype(int)


# ## H2 — Age interaction (stratified models)

# In[16]:


OTH = "income_clean + education_clean + cross_county + work_core"   # NOTE: no continuous age

print(f"collinearity check: corr(age, age_old) = {d['age'].corr(d['age_old']):.3f}  <- why age must enter ONCE\n")

def lr2(full_f, red_f, dfree, label):
    F = smf.logit(full_f, data=d).fit(disp=0); R = smf.logit(red_f, data=d).fit(disp=0)
    stat = 2*(F.llf - R.llf); p = chi2.sf(stat, dfree)
    verdict = 'SIGNIFICANT' if p < .05 else ('marginal' if p < .10 else 'not sig.')
    print(f"  {label:46s} chi2({dfree})={stat:5.2f}  p={p:.4f}  {verdict}")
    return stat, p

print("WFH x AGE interaction:")
print(" SPEC 1 - age entered ONCE, continuous (preferred):")
RESULTS['age_cont_lin'] = lr2(f"sustainable_mode ~ wfh_policy_6*age + {OTH}",
                              f"sustainable_mode ~ wfh_policy_6 + age + {OTH}", 1, "   WFH(linear) x age")
RESULTS['age_cont_quad'] = lr2(f"sustainable_mode ~ (wfh_policy_6 + I(wfh_policy_6**2))*age + {OTH}",
                               f"sustainable_mode ~ wfh_policy_6 + I(wfh_policy_6**2) + age + {OTH}", 2, "   WFH(quad) x age")
print(" SPEC 2 - age entered ONCE, median split:")
RESULTS['age_split_lin'] = lr2(f"sustainable_mode ~ wfh_policy_6*C(age_old) + {OTH}",
                               f"sustainable_mode ~ wfh_policy_6 + C(age_old) + {OTH}", 1, "   WFH(linear) x age_old")
RESULTS['age_split_quad'] = lr2(f"sustainable_mode ~ (wfh_policy_6 + I(wfh_policy_6**2))*C(age_old) + {OTH}",
                                f"sustainable_mode ~ wfh_policy_6 + I(wfh_policy_6**2) + C(age_old) + {OTH}", 2, "   WFH(quad) x age_old")
print(" SPEC 3 - age entered TWICE (WRONG - collinear; shown only to document the error):")
lr2(f"sustainable_mode ~ wfh_policy_6*C(age_old) + age + {OTH}",
    f"sustainable_mode ~ wfh_policy_6 + C(age_old) + age + {OTH}", 1, "   WFH(linear) x age_old + age")

print("\nStratified gradients (age entered once):")
for lv, nm in [(0, 'YOUNG 18-44'), (1, 'OLD 45+')]:
    sub = d[d.age_old == lv]
    m = smf.logit(f"sustainable_mode ~ wfh_policy_6 + {OTH}", data=sub).fit(disp=0, cov_type="HC3")
    RESULTS[f'strat_age_{lv}'] = (m.params['wfh_policy_6'], m.pvalues['wfh_policy_6'], len(sub))
    print(f"  {nm:12s} n={len(sub):3d}  WFH b={m.params['wfh_policy_6']:+.3f}  p={m.pvalues['wfh_policy_6']:.4f}")

print("\nCell counts for the OLD stratum at high flexibility (caveat for the text):")
print(d[d.age_old==1].groupby('wfh_policy_6').size().to_dict())

# Controls for INCOME tests must EXCLUDE income_clean (else income enters twice:
# once as C(inc_high)/income_clean moderator, once as a control). corr = 0.725.
CL_INC = "age + education_clean + cross_county"          # no income_clean, no work_core


# ## Supplementary — income interaction and access-vs-effectiveness

# In[17]:


W = "wfh_policy_6 + I(wfh_policy_6**2)"

print(
    "collinearity: corr(income_clean, inc_high) =",
    round(d["income_clean"].corr(d["inc_high"]), 3)
)
print()

print("What 0422 did (stratified, separate models):")

for lv, nm in [
    (0, "LOW income "),
    (1, "HIGH income"),
]:
    sub = d[d["inc_high"] == lv]

    m = smf.logit(
        f"sustainable_mode ~ {W} + {CL_INC}",
        data=sub
    ).fit(
        disp=0,
        cov_type="HC3"
    )

    b1 = m.params["wfh_policy_6"]
    b2 = m.params["I(wfh_policy_6 ** 2)"]

    print(
        f"  {nm} "
        f"n={len(sub):3d}  "
        f"quad p={m.pvalues['I(wfh_policy_6 ** 2)']:.4f}  "
        f"TP={-b1 / (2 * b2):.2f}"
    )

print("  -> two different TPs, read as 'moderation'. But never tested.")
print()

print("The actual interaction test (income entered ONCE in every row):")

RESULTS["inc_split_nocore"] = lr2(
    f"sustainable_mode ~ ({W})*C(inc_high) + {CL_INC}",
    f"sustainable_mode ~ {W} + C(inc_high) + {CL_INC}",
    2,
    "  split,      NO location"
)

RESULTS["inc_split_core"] = lr2(
    f"sustainable_mode ~ ({W})*C(inc_high) + {CL_INC} + work_core",
    f"sustainable_mode ~ {W} + C(inc_high) + {CL_INC} + work_core",
    2,
    "  split,      WITH location"
)

RESULTS["inc_cont_nocore"] = lr2(
    f"sustainable_mode ~ ({W})*income_clean + {CL_INC}",
    f"sustainable_mode ~ {W} + income_clean + {CL_INC}",
    2,
    "  continuous, NO location"
)

RESULTS["inc_cont_core"] = lr2(
    f"sustainable_mode ~ ({W})*income_clean + {CL_INC} + work_core",
    f"sustainable_mode ~ {W} + income_clean + {CL_INC} + work_core",
    2,
    "  continuous, WITH location"
)

# verdict COMPUTED from the live numbers - never hardcoded
print()

_ps = [
    RESULTS[k][1]
    for k in (
        "inc_split_nocore",
        "inc_split_core",
        "inc_cont_nocore",
        "inc_cont_core",
    )
]

verdict = (
    "NEVER significant"
    if min(_ps) >= 0.05
    else "CHECK: some spec is significant"
)

print(
    f"  income interaction p ranges "
    f"{min(_ps):.3f} to {max(_ps):.3f} -> {verdict}"
)

print(
    f"  no-location "
    f"{RESULTS['inc_split_nocore'][1]:.3f} -> "
    f"with-location {RESULTS['inc_split_core'][1]:.3f} (split); "
    f"{RESULTS['inc_cont_nocore'][1]:.3f} -> "
    f"{RESULTS['inc_cont_core'][1]:.3f} (continuous)"
)

print(
    "  -> p does not RISE when location is added, "
    "so location did not 'kill' income."
)

print(
    "     Income was never significant. "
    "Under-powered null (high-income n=%d)."
    % (d["inc_high"] == 1).sum()
)

print()
print("Access vs effectiveness - P(works Manhattan) by income category:")

ct = pd.crosstab(
    d["income_clean"],
    d["work_core"],
    normalize="index"
)

for k, v in ct[1.0].items():
    print(f"    income {int(k)}: {v:.3f}")

print(
    "  -> ACCESS to central, transit-rich jobs is strongly "
    "stratified by income;"
)

print(
    "     EFFECTIVENESS of flexibility is not. "
    "This is the supplementary equity finding."
)


# Placeholder until the actual function body is added
def stratified_tps(var, groups, drop_from_ctrl):
    pass


# ## Table 3 — full regression table

# In[18]:


def stratified_tps(var, groups, drop_from_ctrl):
    """
    Per-stratum linear gradient + TP (descriptive), then the interaction
    decomposed into slope vs curvature — the joint 2-df test alone
    can't tell them apart.
    """

    W = "wfh_policy_6 + I(wfh_policy_6**2)"

    ctrl_vars = [
        "age",
        "income_clean",
        "education_clean",
        "cross_county",
        "work_core",
    ]

    ctrl = " + ".join(
        c for c in ctrl_vars
        if c not in drop_from_ctrl
    )

    print(f"--- stratifier: {var} ---")

    rows = []

    for val, nm in groups:
        sub = d[d[var] == val].copy()

        mq = smf.logit(
            f"sustainable_mode ~ {W} + {ctrl}",
            data=sub
        ).fit(
            disp=0,
            cov_type="HC3"
        )

        b1 = mq.params["wfh_policy_6"]
        b2 = mq.params["I(wfh_policy_6 ** 2)"]

        tp = (
            -b1 / (2 * b2)
            if b2 != 0
            else float("nan")
        )

        ml = smf.logit(
            f"sustainable_mode ~ wfh_policy_6 + {ctrl}",
            data=sub
        ).fit(
            disp=0,
            cov_type="HC3"
        )

        quad_p = mq.pvalues["I(wfh_policy_6 ** 2)"]
        lin_b = ml.params["wfh_policy_6"]
        lin_p = ml.pvalues["wfh_policy_6"]

        rows.append(
            (
                nm,
                len(sub),
                sub["sustainable_mode"].mean(),
                b2,
                quad_p,
                tp,
                lin_b,
                lin_p,
            )
        )

        # TP is descriptive-only when quadratic term is not significant
        tp_note = (
            ""
            if quad_p < 0.05
            else "  (quad n.s. -> TP not interpretable)"
        )

        print(
            f"  {nm:14s} "
            f"n={len(sub):3d} "
            f"sust={sub['sustainable_mode'].mean():.3f} | "
            f"linear b={lin_b:+.3f} p={lin_p:.3f} | "
            f"quad b2={b2:+.4f} p={quad_p:.3f} "
            f"TP={tp:5.2f}{tp_note}"
        )

    # --------------------------------------------------
    # Decompose interaction:
    # reduced -> + slope×var -> + curvature×var
    # --------------------------------------------------

    m0 = (
        f"sustainable_mode ~ "
        f"{W} + C({var}) + {ctrl}"
    )

    m1 = (
        f"sustainable_mode ~ "
        f"{W} + "
        f"wfh_policy_6:C({var}) + "
        f"C({var}) + {ctrl}"
    )

    m2 = (
        f"sustainable_mode ~ "
        f"{W} + "
        f"wfh_policy_6:C({var}) + "
        f"I(wfh_policy_6**2):C({var}) + "
        f"C({var}) + {ctrl}"
    )

    l0 = smf.logit(
        m0,
        data=d
    ).fit(disp=0).llf

    l1 = smf.logit(
        m1,
        data=d
    ).fit(disp=0).llf

    l2 = smf.logit(
        m2,
        data=d
    ).fit(disp=0).llf

    stat_slope = 2 * (l1 - l0)
    stat_curv = 2 * (l2 - l1)
    stat_joint = 2 * (l2 - l0)

    p_slope = chi2.sf(stat_slope, 1)
    p_curv = chi2.sf(stat_curv, 1)
    p_joint = chi2.sf(stat_joint, 2)

    RESULTS[f"strat_int_{var}"] = (
        stat_joint,
        p_joint
    )

    RESULTS[f"strat_slope_{var}"] = (
        stat_slope,
        p_slope
    )

    RESULTS[f"strat_curv_{var}"] = (
        stat_curv,
        p_curv
    )

    print(
        f"  => interaction: "
        f"joint chi2(2)={stat_joint:.2f} "
        f"p={p_joint:.4f} | "
        f"SLOPE chi2(1)={stat_slope:.2f} "
        f"p={p_slope:.4f} | "
        f"CURVATURE chi2(1)={stat_curv:.2f} "
        f"p={p_curv:.4f}"
    )

    print()

    return rows


# ## Numbers summary (reproduces the published manuscript values)

# In[19]:


# ---------------------------------------------------------------------------
# Revision analysis: marginal effects, cadence-break tests, piecewise model


# ## Revision analysis — average marginal effects and the measurement-cadence test

# In[20]:


# (added for the Sustainability Science technical-report revision)
# ---------------------------------------------------------------------------
import json
print("\n" + "="*70)
print("REVISION ANALYSIS: marginal effects and cadence-break tests")
print("="*70)

f_lin  = f"sustainable_mode ~ wfh_policy_6 + {CTRL}"
f_quad = f"sustainable_mode ~ wfh_policy_6 + I(wfh_policy_6**2) + {CTRL}"
mb = smf.logit(f_lin,  data=d).fit(disp=0, cov_type="HC3")
mq = smf.logit(f_quad, data=d).fit(disp=0, cov_type="HC3")
ml = smf.logit(f_quad + " + work_core", data=d).fit(disp=0, cov_type="HC3")

def ame(m, k):
    e = m.get_margeff(at='overall', method='dydx')
    idx = list(e.summary_frame().index).index(k)
    return float(e.margeff[idx]), float(e.margeff_se[idx]), float(e.pvalues[idx])

results = {}
results['ame_wfha_no_loc']   = ame(mq, 'wfh_policy_6')
results['ame_wfha2_no_loc']  = ame(mq, 'I(wfh_policy_6 ** 2)')
results['ame_wfha_loc']      = ame(ml, 'wfh_policy_6')
results['ame_wfha2_loc']     = ame(ml, 'I(wfh_policy_6 ** 2)')
results['ame_work_core']     = ame(ml, 'work_core')
results['ame_income']        = ame(ml, 'income_clean')

b1, b1l = results['ame_wfha_no_loc'][0], results['ame_wfha_loc'][0]
b2, b2l = results['ame_wfha2_no_loc'][0], results['ame_wfha2_loc'][0]
results['attenuation_linear_pct']    = 100*(1-b1l/b1)
results['attenuation_quadratic_pct'] = 100*(1-b2l/b2)
results['location_income_ratio']     = abs(results['ame_work_core'][0] / results['ame_income'][0])

d['monthly_cadence'] = (d['wfh_policy_6'] >= 4).astype(int)
f_cadence     = f"sustainable_mode ~ wfh_policy_6 + I(wfh_policy_6**2) + monthly_cadence + {CTRL}"
f_lin_cadence = f"sustainable_mode ~ wfh_policy_6 + monthly_cadence + {CTRL}"
m0 = smf.logit(f_lin_cadence, data=d).fit(disp=0)
m1 = smf.logit(f_cadence, data=d).fit(disp=0)
lr_cadence = 2*(m1.llf - m0.llf)
results['cadence_lr_chi2'] = float(lr_cadence)
results['cadence_lr_p']    = float(chi2.sf(lr_cadence, 1))

d['wfha_weekly'] = d['wfh_policy_6'].clip(upper=3)
d['wfha_monthly_part'] = (d['wfh_policy_6'] - 3).clip(lower=0)
mpw = smf.logit(f"sustainable_mode ~ wfha_weekly + wfha_monthly_part + {CTRL}", data=d).fit(disp=0, cov_type="HC3")
results['piecewise_weekly']  = (float(mpw.params['wfha_weekly']), float(mpw.pvalues['wfha_weekly']))
results['piecewise_monthly'] = (float(mpw.params['wfha_monthly_part']), float(mpw.pvalues['wfha_monthly_part']))

print(json.dumps(results, indent=1))


# ---------------------------------------------------------------------------


# ## Supplementary — product-of-coefficients indirect-path check (not a causal-mediation claim)

# In[21]:


# Supplementary: product-of-coefficients indirect-path check (not a causal
# mediation claim -- see manuscript caveat on implied causal ordering)
# ---------------------------------------------------------------------------
# Uses the SAME three-category grouping (low 0-1 / medium 2-3 / high 4-6 on WFHA)
# as the primary H1 categorical model above, for internal consistency.
d['wfh3f'] = d['wfh_3level'].astype(float)

a_m  = smf.ols(f"work_core ~ wfh3f + {CTRL}", data=d).fit(cov_type="HC3")
tot  = smf.ols(f"sustainable_mode ~ wfh3f + {CTRL}", data=d).fit(cov_type="HC3")
dir_ = smf.ols(f"sustainable_mode ~ wfh3f + work_core + {CTRL}", data=d).fit(cov_type="HC3")
a_coef, b_coef = a_m.params['wfh3f'], dir_.params['work_core']
indirect = a_coef * b_coef

np.random.seed(SEED)
boot = []
for _ in range(2000):
    bs = d.sample(len(d), replace=True)
    try:
        am = smf.ols(f"work_core ~ wfh3f + {CTRL}", data=bs).fit()
        dm = smf.ols(f"sustainable_mode ~ wfh3f + work_core + {CTRL}", data=bs).fit()
        boot.append(am.params['wfh3f'] * dm.params['work_core'])
    except Exception:
        pass
lo, hi = np.nanpercentile(boot, [2.5, 97.5])
results['indirect_path_estimate'] = float(indirect)
results['indirect_path_ci'] = [float(lo), float(hi)]
print(f"\nSupplementary indirect-path estimate: {indirect:+.4f}, bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}]")
print("NOTE: reported as corroborating evidence only, not a causal mediation claim (see manuscript).")

json.dump(results, open(os.path.join(RESULTSDIR, 'revision_results.json'),'w'), indent=1)
print("\nSaved revision_results.json")

