# Breaking the Daily Routine (NYC WFH and Sustainable Commuting)

Code and materials to reproduce every reported value, table, and figure in:

> [AUTHORS]. [YEAR]. *Breaking the Daily Routine: Habit Disruption, Vehicle Ownership, and the
> Impact of Work-from-Home on Sustainable Commuting*. Submitted to *Sustainability Science*.

Repository: <https://github.com/zhaxinge/nyc-wfh>

## Repository structure

```
scripts/    replication.py          — the full analysis for manuscript v4
            robustness_2024.py      — robustness check: replication with the 2024 CMS
notebooks/  replication.ipynb       — the same analysis as a notebook (committed with outputs)
figures/    fig2_pooled_wfha.png    — Figure 2 (written by the script)
            fig3_group_wfha.png     — Figure 3 (written by the script)
results/    results_v4.json         — every reported number (written by the script)
            robustness_2024.json    — results of the 2024 robustness check
archive/    technical-report/       — code, figure and results for the previous revision
data/raw/   (not tracked) NYC Citywide Mobility Survey Person file — see below
```

## What is reproduced

| Script section | Manuscript |
|---|---|
| 1 | Analytic sample (N = 715), Tables 1.1 and 1.2 |
| 2 | Table 2, Figure 2, leave-one-out, bootstrap vertex, three-level checks (§4.1) |
| 3 | Table 3: mediation through household vehicles (§4.2) |
| 4 | Table 4 (age groups), Table 5 (workplace groups), Figure 3 (§4.3–4.4) |
| 5 | Probability-weighted sensitivity models (§4.5) |
| 6 | Appendix Tables A.1–A.5 |

## Data

The analysis uses the 2022 NYC Citywide Mobility Survey (CMS) **Person** table
(6,886 respondents; analytic sample N = 715), published by NYC DOT on NYC Open Data:

<https://data.cityofnewyork.us/Transportation/Citywide-Mobility-Survey-Person-2022/7qdz-u9hr>

Export it as CSV and save it as:

```
data/raw/Citywide_Mobility_Survey_-_Person_HOUSEHOLD.csv
```

Raw data are not redistributed here. To use a different location, set `WFH_CSV=/path/to/file.csv`.

## How to reproduce

```bash
pip install -r requirements.txt
python scripts/replication.py
# or: jupyter nbconvert --to notebook --execute notebooks/replication.ipynb
```

Runtime is about 7 minutes (bootstrap steps: 1,000 resamples, fixed seeds). Figures go to `figures/` and numeric results
to `results/`. Override these locations with `WFH_FIGDIR` and `WFH_RESULTSDIR`.

Every reported value is compared with the manuscript by a `CHECK` line. The run continues past a
mismatch, lists all mismatches at the end, and then stops with an assertion error.
All 345 checks pass with the pinned package versions. Bootstrap interval bounds are
checked to within Monte Carlo error, because the manuscript's draws came from a different random stream.

## Robustness check: 2024 Citywide Mobility Survey

`scripts/robustness_2024.py` asks whether the main pattern holds in the 2024 CMS. The 2024 survey
does not ask about the employer's in-office requirement, so WFHA cannot be rebuilt. Both waves do record
the respondent's **actual telework frequency**. The script puts it on one 0–6 scale (0 = never, including
daily commuters, who were not asked; 6 = 5+ days a week; the 2024 "2 days" and "3 days" answers are merged
into the 2022 "2–3 days" category) and estimates the same models in 2022 and 2024:

1. Samples and the harmonized scale
2. Main association: linear, quadratic, + Manhattan, + borough and industry, survey-weighted
3. Three-level contrasts (none / up to 1 day a week / 2+ days): logit, linear probability, weighted, Firth
4. Robustness grid: alternative exposure codings, controls and samples, pooled and in four groups
5. Telework × year interaction tests on the pooled 2022 + 2024 data
6. Heterogeneity by age and workplace location
7. Mediation through household vehicles, with bootstrap intervals
8. Weekly car-commute days and commute days
9. Reported change in household vehicles

The 2022 sample follows the corrected definition of the revised manuscript (N = 1,305: daily commuters
included, "other" mode excluded), not the v4 definition used by `replication.py`. No values are checked
against the manuscript, because these results are not yet reported there.

Additional inputs (NYC Open Data, export as CSV):

- Person 2024: <https://data.cityofnewyork.us/Transportation/Citywide-Mobility-Survey-Person-2024/i2im-iqtt>
  → `data/raw/Citywide_Mobility_Survey_-_Person_2024.csv` (or set `WFH_CSV_2024_PERSON`)
- Household 2024: <https://data.cityofnewyork.us/Transportation/Citywide-Mobility-Survey-Household-2024/5mb3-padx>
  → `data/raw/Citywide_Mobility_Survey_-_Household_2024.csv` (or set `WFH_CSV_2024_HOUSEHOLD`)

```bash
python scripts/robustness_2024.py     # about 10 minutes with 1,000 bootstrap resamples (WFH_BOOT)
```

## Software

Package versions are pinned in `requirements.txt` (Python 3.11). The original analysis ran on
Google Colab.

## License

Code: MIT (see `LICENSE`).

## Citation

See `CITATION.cff`. An archived release with a DOI: [ZENODO DOI — TO BE COMPLETED].
