# Replication Package — WFH Policy and Sustainable Commuting (NYC)

Code and materials to reproduce every reported value, table, and figure in:

> [AUTHORS]. [YEAR]. *[MANUSCRIPT TITLE]*. Submitted to *Sustainability Science*.

## Repository structure

```
notebooks/   replication_sustsci.ipynb   — end-to-end replication notebook
scripts/     replication_sustsci.py      — same analysis as a plain script
             make_figures.py             — figure-generation code
figures/     Figure_1.png, Figure_2.png  — generated figures
data/raw/    (not tracked) NYC Citywide Mobility Survey extract — see below
data/processed/  intermediate files written by the scripts
results/     model outputs / tables written by the scripts
```

## Data

The analysis uses the NYC Citywide Mobility Survey (CMS) person-level data (analytic sample N = 715),
available from NYC Open Data: [DATASET URL — TO BE COMPLETED].

Download the CSV and save it as `data/raw/[FILENAME].csv`. Raw data are not redistributed here.

## How to reproduce

```bash
pip install -r requirements.txt
python scripts/replication_sustsci.py      # or run notebooks/replication_sustsci.ipynb
python scripts/make_figures.py
```

## Software

Python [VERSION]; package versions pinned in `requirements.txt`.

## License

Code: MIT (see `LICENSE`).

## Citation

See `CITATION.cff`. An archived release with a DOI: [ZENODO DOI — TO BE COMPLETED].
