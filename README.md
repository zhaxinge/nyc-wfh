# Replication Package — Work-from-Home Flexibility and Sustainable Commuting (NYC)

Code and materials to reproduce the reported values in:

> [AUTHORS]. *Work-from-Home Flexibility and Sustainable Commuting: A Technical Report on Diminishing
> Returns, Age Heterogeneity, and Attenuation by Workplace Location*. Submitted to *Sustainability Science*.

Repository: <https://github.com/zhaxinge/nyc-wfh>

## Repository structure

```
notebooks/  replication_sustsci.ipynb  — end-to-end replication notebook (committed with outputs)
scripts/    replication_sustsci.py     — the same analysis as a plain Python script
results/    revision_results.json      — AMEs, attenuation, cadence test, piecewise slopes, indirect path
            descriptives_weighted_unweighted.csv — weighted and unweighted statistics, every variable
figures/    fig1_turning_point.png     — fitted quadratic vs. observed shares (diagnostic figure)
data/raw/   (not tracked) NYC Citywide Mobility Survey Person file — see below
```

## Data

The analysis uses the 2022 NYC Citywide Mobility Survey (CMS) **Person** table
(NYC Department of Transportation 2022; 6,886 respondents; analytic sample N = 715,
about 1.23 million when weighted), published on NYC Open Data:

<https://data.cityofnewyork.us/Transportation/Citywide-Mobility-Survey-Person-2022/7qdz-u9hr>

Export it as CSV and save it as:

```
data/raw/Citywide_Mobility_Survey_-_Person_HOUSEHOLD.csv
```

Raw data are not redistributed here. To use a different location, set `WFH_CSV=/path/to/file.csv`.

## Key variables

| Name in code       | Manuscript       | Definition |
|--------------------|------------------|------------|
| `sustainable_mode` | sustainable_mode | 1 = primary commute by walking, bus/shuttle, bicycle, rail, micromobility, or ferry; 0 = private vehicle, other motor vehicle, ride-hailing |
| `wfh_policy_6`     | WFHA             | Employer in-office requirement, reverse-coded 0–6 (0 = required 5+ days/week … 6 = never required). Levels 0–3 use a weekly cadence and 4–6 a monthly cadence, so the scale is non-interval |
| `wfh_3level`       | low / medium / high | WFHA 0–1 / 2–3 / 4–6 |
| `work_core`        | work_manhattan   | 1 = works in CMS work zones 3–4 (Manhattan; 71.9% of the sample) |
| `age`, `education_clean`, `income_clean`, `cross_county` | controls | Coded CMS age, education and income; cross-county commute based on home and work county FIPS |
| `w`, `w_norm`      | survey weight    | CMS person weight (normalized to mean 1 in the weighted models) |

## Where each result is produced

Section names refer to headings in the notebook.

| Manuscript result | Notebook section |
|---|---|
| Sample N = 715, weighted N ≈ 1.23 M, 71.9% work in Manhattan | *Load data and build the analytic sample* |
| Weighted vs. unweighted descriptives (Methods) | *Manuscript values not printed above* → `results/descriptives_weighted_unweighted.csv` |
| Quadratic β = 0.837 / −0.105, robust SEs, vertex 3.97 | §4.1b; SEs in *Manuscript values not printed above* |
| Leave-one-cell-out (drop WFHA = 0: p = 0.098) | §4.1b |
| Bootstrap: 74.3% significant, vertex 95% CI [3.22, 7.02] (1,000 resamples) | §4.1c |
| Categorical AMEs: low −0.051 (p = 0.035), high +0.004 (p = 0.91) | *Manuscript values not printed above* |
| Piecewise slopes +0.58 / −0.20; cadence-dummy LR χ²(1) = 2.64, p = 0.10 | *Revision analysis* → `revision_results.json` |
| WFHA × age LR χ²(1) = 8.31; strata β = +0.31 / −0.08; older-stratum cell counts | H2 — Age interaction |
| Curvature × age LR χ²(1) = 0.00, p = 0.96 | *Manuscript values not printed above* |
| work_manhattan β = 2.442 (SE 0.293); 36% log-odds reductions | §4.2; *Manuscript values not printed above* |
| AME attenuation 46%; location/income AME ratio ≈ 28 | *Revision analysis* → `revision_results.json` |
| WFHA × location joint χ²(2) = 0.91, p = 0.63 | *Manuscript values not printed above* |
| Indirect path 0.023, 95% CI 0.005–0.042 (2,000 resamples) | *Supplementary — product-of-coefficients* |
| Weighted sensitivity models | *Weighted, location-adjusted sensitivity model* |

## How to reproduce

```bash
pip install -r requirements.txt
python scripts/replication_sustsci.py
# or: jupyter nbconvert --to notebook --execute notebooks/replication_sustsci.ipynb
```

Runtime is about 2 minutes. The bootstrap steps use fixed seeds. Figures go to `figures/` and numeric
results to `results/`. Override these locations with `WFH_FIGDIR` and `WFH_RESULTSDIR`.

The notebook checks itself against the published values, for example N = 715 and the original-coding
turning point of 3.880. It stops with an assertion error if one does not reproduce.

## Software

Verified with Python 3.11; package versions are pinned in `requirements.txt`. The original analysis
ran on Google Colab. Re-running with the pinned versions reproduces every printed value
(differences appear only in the 15th significant digit).

## License

Code: MIT (see `LICENSE`).

## Citation

See `CITATION.cff`. An archived release with a DOI: [ZENODO DOI — TO BE COMPLETED].
