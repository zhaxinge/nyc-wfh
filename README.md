# Replication Package — WFH Policy and Sustainable Commuting (NYC)

Code and materials to reproduce every reported value, table, and figure in:

> [AUTHORS]. [YEAR]. *[MANUSCRIPT TITLE]*. Submitted to *Sustainability Science*.

Repository: <https://github.com/zhaxinge/nyc-wfh>

## Repository structure

```
notebooks/  replication_sustsci.ipynb  — end-to-end replication notebook (committed with outputs)
scripts/    replication_sustsci.py     — the same analysis as a plain Python script
figures/    fig1_turning_point.png     — figure written by the notebook/script
results/    revision_results.json      — marginal effects, cadence tests, indirect-path check
data/raw/   (not tracked) NYC Citywide Mobility Survey Person file — see below
```

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
python scripts/replication_sustsci.py
# or: jupyter nbconvert --to notebook --execute notebooks/replication_sustsci.ipynb
```

Runtime is about 2 minutes (bootstrap steps: 1,000 and 2,000 resamples, fixed seeds).
Figures go to `figures/` and numeric results to `results/`. Override these locations with
`WFH_FIGDIR` and `WFH_RESULTSDIR`.

The notebook checks itself against the published values, for example the original-coding
turning point (3.880) and N = 715. It stops with an assertion error if one does not reproduce.

## Software

Verified with Python 3.11; package versions are pinned in `requirements.txt`. The original analysis
ran on Google Colab. Re-running with the pinned versions reproduces every printed value
(differences appear only in the 15th significant digit).

## License

Code: MIT (see `LICENSE`).

## Citation

See `CITATION.cff`. An archived release with a DOI: [ZENODO DOI — TO BE COMPLETED].
