# Breaking the Daily Routine (NYC WFH and Sustainable Commuting)

Code and materials to reproduce every reported value, table, and figure in:

> [AUTHORS]. [YEAR]. *Breaking the Daily Routine: Habit Disruption, Vehicle Ownership, and the
> Impact of Work-from-Home on Sustainable Commuting*. Submitted to *Sustainability Science*.

Repository: <https://github.com/zhaxinge/nyc-wfh>

## Repository structure

```
scripts/    replication.py          — the full analysis for manuscript v4
notebooks/  replication.ipynb       — the same analysis as a notebook (committed with outputs)
figures/    fig2_pooled_wfha.png    — Figure 2 (written by the script)
            fig3_group_wfha.png     — Figure 3 (written by the script)
results/    results_v4.json         — every reported number (written by the script)
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

## Software

Package versions are pinned in `requirements.txt` (Python 3.11). The original analysis ran on
Google Colab.

## License

Code: MIT (see `LICENSE`).

## Citation

See `CITATION.cff`. An archived release with a DOI: [ZENODO DOI — TO BE COMPLETED].
