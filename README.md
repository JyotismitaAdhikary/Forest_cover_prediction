# Assam Forest Cover Analysis

A machine learning pipeline for monitoring forest cover change in Assam, India (2015–2023) using satellite data from Google Earth Engine.

---

## Overview

This project combines multi-source remote sensing data — Landsat 8 imagery, ERA5 climate reanalysis, Hansen Global Forest Change data, and SRTM topography — to classify pixels as forest or non-forest, then tracks how Assam's forest cover has changed over time. Two ML classifiers (Random Forest and LightGBM) are trained on annual pixel samples and evaluated using a temporal hold-out strategy.

There are **two ways to run it**:

- **A narrated Jupyter notebook** (`forest_cover_classification.ipynb`) — a self-contained, step-by-step walkthrough that runs end-to-end **without a Google Earth Engine account**. It includes the full Earth Engine methodology *and* a synthetic data sample with the same structure, so the whole pipeline (cleaning → features → models → charts) runs anywhere.
- **The full command-line pipeline** (`main.py`) — pulls live data from Google Earth Engine and runs the complete workflow.

---

## Features

- **Multi-source data fusion** — NDVI/EVI from Landsat 8, temperature and precipitation from ERA5, elevation and slope from SRTM, and binary forest labels from Hansen GFC
- **Temporal feature engineering** — lagged NDVI, 3-year rolling mean, and year-over-year delta
- **Two classifiers** — Random Forest and LightGBM, both with `class_weight="balanced"` to handle label imbalance
- **Temporal train/test split** — trains on earlier years, tests on held-out years (default: 2022–2023)
- **Annual forest cover trend** — applies the best model across all years to estimate % forest cover per year
- **Narrated notebook** — runs without an Earth Engine account using a synthetic sample (real GEE code included)
- **CLI pipeline** — fully automated end-to-end run, saves all outputs to `outputs/`

---

## Data Sources

| Source | Dataset | Features |
|---|---|---|
| Landsat 8 (USGS) | `LANDSAT/LC08/C02/T1_L2` | NDVI, EVI |
| ERA5-Land (ECMWF) | `ECMWF/ERA5_LAND/MONTHLY_AGGR` | Mean temperature, total precipitation |
| Hansen GFC | `UMD/hansen/global_forest_change_2023_v1_11` | Forest/non-forest labels, loss year |
| SRTM (USGS) | `USGS/SRTMGL1_003` | Elevation, slope |
| FAO GAUL | `FAO/GAUL/2015/level1` | Assam boundary |

---

## Project Structure

```
├── forest_cover_classification.ipynb  # Narrated notebook (self-contained demo)
├── main.py                # Full-pipeline entry point
├── config.py              # Shared configuration
├── data_collection.py     # GEE data fetching and sampling
├── preprocessing.py       # Cleaning, feature engineering, train/test split
├── models.py              # Model definitions and serialization
├── evaluate.py            # Metrics and annual forest cover computation
├── visualize.py           # Matplotlib figure helpers
├── requirements.txt       # Dependencies
└── outputs/               # Generated when you run the pipeline (CSVs, plots, models)
```

---

## Quickstart

### Option A — Notebook (quickest, no GEE account needed)

The notebook runs the full pipeline on a synthetic sample, so you can see everything work without any setup or credentials.

```bash
pip install -r requirements.txt
pip install notebook            # if you don't already have Jupyter
jupyter notebook forest_cover_classification.ipynb
```

Then choose **Kernel → Restart & Run All**. To run it on **real** data, authenticate with Earth Engine (see Option B) and swap the synthetic-sample cell for the `fetch_samples_for_year(...)` calls shown in the notebook.

### Option B — Full pipeline on live GEE data

**Prerequisites**
- A [Google Earth Engine](https://earthengine.google.com/) account
- Python 3.10+

```bash
# Install dependencies
pip install -r requirements.txt

# Authenticate with GEE (one-time)
earthengine authenticate

# Run the full pipeline
python main.py
```

On first run, data is fetched from GEE and cached to `outputs/raw_data.csv`. Subsequent runs load from cache automatically.

**Outputs saved to `outputs/`:**
- `processed_data.csv` — cleaned feature matrix
- `forest_cover_trend.csv` — predicted % forest cover per year
- `model_comparison.csv` — ROC-AUC and F1 for each model
- `random_forest.joblib`, `lightgbm.joblib` — serialized trained models
- `forest_cover_trend.png`, `roc_curves.png`, `ndvi_distribution.png`, confusion matrix and feature importance PNGs

---

## Configuration

All parameters are set in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `region` | `"Assam"` | GAUL ADM1 region name |
| `start_year` | `2015` | First year of data collection |
| `end_year` | `2023` | Last year of data collection |
| `cloud_threshold` | `30` | Max cloud cover % for Landsat filtering |
| `forest_threshold` | `30` | Min tree cover % for Hansen forest label |
| `scale` | `500` | Sampling scale in metres |
| `seed` | `42` | Random seed for reproducibility |

---

## Models

Both models are wrapped in a `StandardScaler → Classifier` sklearn `Pipeline`.

**Random Forest** — 200 trees, max depth 15, balanced class weights.

**LightGBM** — 500 estimators, learning rate 0.05, 63 leaves, subsample 0.8, balanced class weights.

Feature importance is extracted from the fitted classifier step and visualized as a horizontal bar chart.

*Note: the `StandardScaler` is kept for consistency, though tree-based models are scale-invariant and don't require it.*

---

## Evaluation

Models are evaluated on held-out test years (default: 2022, 2023) with:

- ROC-AUC
- F1 score
- Precision and recall per class (Forest / Non-Forest)
- Confusion matrix
- ROC curve comparison

---

## Requirements

```
earthengine-api==0.1.401
pandas==2.1.4
numpy==1.26.4
scikit-learn==1.5.0
lightgbm==4.3.0
matplotlib==3.9.0
joblib==1.4.2
```

The notebook additionally needs Jupyter (`pip install jupyte notebook`).

---
