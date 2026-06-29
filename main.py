"""
Assam Forest Cover Analysis — Command Line Version
====================================================
Run:  python main.py

Steps:
  1. Authenticates and connects to Google Earth Engine
  2. Fetches Landsat + ERA5 + Hansen + SRTM data for Assam (2015–2023)
  3. Cleans data and engineers temporal features
  4. Trains Random Forest and LightGBM models
  5. Evaluates models and prints metrics
  6. Saves plots, CSVs, and trained models to ./outputs/
"""

import os
import json
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, roc_curve,
)
import lightgbm as lgb
import ee

warnings.filterwarnings("ignore")

# ── Output directory ───────────────────────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
CONFIG = {
    "region": "Assam",
    "gaul_field": "ADM1_NAME",
    "start_year": 2015,
    "end_year": 2023,
    "cloud_threshold": 30,
    "forest_threshold": 30,
    "scale": 500,
    "seed": 42,
    "samples_per_class": 1500,
    "test_years": [2022, 2023],
}

FEATURE_COLS = ["NDVI", "EVI", "mean_temp", "total_precip", "elevation", "slope"]
TARGET_COL = "label"


# ══════════════════════════════════════════════════════════════════════════════
# GEE INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════

def initialize_gee():
    """
    Tries standard GEE auth first (works if you've run `earthengine authenticate`
    on your machine before). Falls back to interactive authenticate if needed.
    """
    print("Connecting to Google Earth Engine...")
    try:
        ee.Initialize(project='assamforestcover')
        print("Connected to GEE.")
    except Exception:
        print("Not authenticated. Launching browser auth...")
        ee.Authenticate()
        ee.Initialize()
        print("Connected to GEE.")


# ══════════════════════════════════════════════════════════════════════════════
# DATA COLLECTION
# ══════════════════════════════════════════════════════════════════════════════

def get_assam_boundary():
    return (
        ee.FeatureCollection("FAO/GAUL/2015/level1")
        .filter(ee.Filter.eq(CONFIG["gaul_field"], CONFIG["region"]))
    )


def mask_landsat_clouds(image):
    qa = image.select("QA_PIXEL")
    cloud_mask = qa.bitwiseAnd(1 << 3).eq(0)
    shadow_mask = qa.bitwiseAnd(1 << 4).eq(0)
    return image.updateMask(cloud_mask.And(shadow_mask))


def compute_indices(image):
    ndvi = image.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")
    evi = image.expression(
        "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
        {
            "NIR": image.select("SR_B5"),
            "RED": image.select("SR_B4"),
            "BLUE": image.select("SR_B2"),
        },
    ).rename("EVI")
    return image.addBands([ndvi, evi])


def get_annual_composite(year, region):
    return (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(region)
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .filter(ee.Filter.lt("CLOUD_COVER", CONFIG["cloud_threshold"]))
        .map(mask_landsat_clouds)
        .map(compute_indices)
        .median()
        .select(["NDVI", "EVI"])
        .clip(region)
    )


def get_hansen_labels(year, region):
    hansen = ee.Image("UMD/hansen/global_forest_change_2023_v1_11")
    tree_cover = hansen.select("treecover2000")
    loss_year = hansen.select("lossyear")

    # A pixel is "Forest" in a given year if:
    #   - It had >= forest_threshold% tree cover in 2000, AND
    #   - It had not yet been lost by this year (loss_year == 0 means no loss ever,
    #     loss_year > (year-2000) means loss happened AFTER this year)
    yr_offset = year - 2000
    was_forested = tree_cover.gte(CONFIG["forest_threshold"])
    not_yet_lost = loss_year.eq(0).Or(loss_year.gt(yr_offset))
    forest_mask = was_forested.And(not_yet_lost)

    label = forest_mask.rename("label").clip(region)
    loss_band = loss_year.rename("loss_year").clip(region)
    return label, loss_band


def get_climate_data(year, region):
    era5 = (
        ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .filterBounds(region)
        .select(["temperature_2m", "total_precipitation_sum"])
    )
    mean_temp = era5.select("temperature_2m").mean().rename("mean_temp")
    total_precip = era5.select("total_precipitation_sum").sum().rename("total_precip")
    return mean_temp.addBands(total_precip).clip(region)


def get_topographic_data(region):
    dem = ee.Image("USGS/SRTMGL1_003").clip(region)
    elevation = dem.select("elevation").rename("elevation")
    slope = ee.Terrain.slope(dem).rename("slope")
    return elevation.addBands(slope)


def fetch_samples_for_year(region, year):
    print(f"  Fetching {year}...", end=" ", flush=True)
    label, loss_band = get_hansen_labels(year, region)
    topo = get_topographic_data(region)
    composite = get_annual_composite(year, region)
    climate = get_climate_data(year, region)

    stacked = (
        composite
        .addBands(climate)
        .addBands(topo)
        .addBands(label)
        .addBands(loss_band)
    )

    samples = stacked.stratifiedSample(
        numPoints=CONFIG["samples_per_class"],
        classBand="label",
        region=region.geometry(),
        scale=CONFIG["scale"],
        seed=CONFIG["seed"],
        geometries=True,
    )

    features = samples.getInfo()
    rows = []
    for f in features["features"]:
        props = f["properties"]
        # Extract coordinates from geometry if available
        geom = f.get("geometry")
        if geom and geom.get("coordinates"):
            props["longitude"] = geom["coordinates"][0]
            props["latitude"] = geom["coordinates"][1]
        rows.append(props)
    df = pd.DataFrame(rows)
    df["year"] = year
    print(f"{len(df)} samples")
    return df


def fetch_all_data():
    # Check for cached CSV to avoid re-fetching
    cache_path = "outputs/raw_data.csv"
    if os.path.exists(cache_path):
        cached = pd.read_csv(cache_path)
        if "longitude" in cached.columns and "latitude" in cached.columns:
            # Also invalidate if only one class present (bad labels)
            if cached["label"].nunique() > 1:
                print(f"Loading cached data from {cache_path}")
                return cached
            else:
                print("Cached data has only one class — re-fetching from GEE with fixed labels...")
                os.remove(cache_path)
        else:
            print("Cached data missing coordinates — re-fetching from GEE...")
            os.remove(cache_path)

    initialize_gee()
    region = get_assam_boundary()
    all_dfs = []

    print(f"\nFetching data for years {CONFIG['start_year']}–{CONFIG['end_year']}:")
    for year in range(CONFIG["start_year"], CONFIG["end_year"] + 1):
        try:
            df_year = fetch_samples_for_year(region, year)
            all_dfs.append(df_year)
        except Exception as e:
            print(f"  WARNING: Could not fetch {year}: {e}")

    if not all_dfs:
        raise RuntimeError("No data fetched. Check your GEE connection and permissions.")

    raw_df = pd.concat(all_dfs, ignore_index=True)
    raw_df.to_csv(cache_path, index=False)
    print(f"\nRaw data saved to {cache_path}")
    return raw_df


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def clean_data(df):
    initial = len(df)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    df = df[df["NDVI"].between(-1, 1)]
    df = df[df["EVI"].between(-3, 3)]
    df = df[df["elevation"] >= 0]
    dropped = initial - len(df)
    print(f"Cleaning: dropped {dropped:,} invalid rows, {len(df):,} remaining.")
    return df.reset_index(drop=True)


def add_temporal_features(df):
    # Round coordinates to ~500m grid to group nearby pixels as the same location
    has_coords = "longitude" in df.columns and "latitude" in df.columns

    if has_coords:
        df["lon_r"] = df["longitude"].round(2)
        df["lat_r"] = df["latitude"].round(2)
        grp = ["lon_r", "lat_r"]
        sort_cols = ["lon_r", "lat_r", "year"]
    else:
        # Fallback: treat each row index as its own location (no spatial grouping)
        print("  Warning: no coordinates found — temporal lag features will be NaN.")
        df["_loc"] = df.index // (CONFIG["end_year"] - CONFIG["start_year"] + 1)
        grp = ["_loc"]
        sort_cols = ["year"]

    df = df.sort_values(sort_cols).reset_index(drop=True)
    df["NDVI_lag1"] = df.groupby(grp)["NDVI"].shift(1)
    df["NDVI_roll3"] = (
        df.groupby(grp)["NDVI"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )
    df["NDVI_delta"] = df["NDVI"] - df["NDVI_lag1"]

    # Drop helper columns
    for col in ["lon_r", "lat_r", "_loc"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    extended = FEATURE_COLS + ["NDVI_lag1", "NDVI_roll3", "NDVI_delta"]
    return df, extended


def temporal_split(df, feature_cols):
    test_years = CONFIG["test_years"]
    train_df = df[~df["year"].isin(test_years)]
    test_df = df[df["year"].isin(test_years)]
    available = [c for c in feature_cols if c in df.columns]

    X_train = train_df[available].values
    y_train = train_df[TARGET_COL].astype(int).values
    X_test = test_df[available].values
    y_test = test_df[TARGET_COL].astype(int).values

    train_yrs = sorted(train_df["year"].unique().tolist())
    print(f"Train years: {train_yrs}  |  Test years: {test_years}")
    print(f"Train samples: {len(X_train):,}  |  Test samples: {len(X_test):,}")
    print(f"Train class dist: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"Test  class dist: {dict(zip(*np.unique(y_test,  return_counts=True)))}")
    return X_train, X_test, y_train, y_test, available


# ══════════════════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════════════════

def build_random_forest():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_leaf=10,
            class_weight="balanced", n_jobs=-1, random_state=CONFIG["seed"],
        )),
    ])


def build_lightgbm():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=63,
            max_depth=10, min_child_samples=20, subsample=0.8,
            colsample_bytree=0.8, class_weight="balanced",
            n_jobs=-1, random_state=CONFIG["seed"], verbose=-1,
        )),
    ])


def get_feature_importance(model, feature_names):
    clf = model.named_steps["clf"]
    return (
        pd.DataFrame({"feature": feature_names, "importance": clf.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)

    if proba.shape[1] < 2:
        print(f"  WARNING: Only one class in predictions for {name}. Skipping AUC.")
        f1 = round(f1_score(y_test, y_pred, zero_division=0), 4)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, labels=[0, 1], target_names=["Non-Forest", "Forest"], zero_division=0)
        print(f"\n{'='*50}\n  {name}\n{'='*50}")
        print(f"  ROC-AUC : N/A\n  F1 Score: {f1}\n\n{report}")
        return {"roc_auc": None, "f1": f1, "confusion_matrix": cm,
                "report": report, "y_pred": y_pred, "y_proba": y_pred.astype(float)}

    y_proba = proba[:, 1]
    auc = round(roc_auc_score(y_test, y_proba), 4)
    f1 = round(f1_score(y_test, y_pred), 4)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Non-Forest", "Forest"])

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  ROC-AUC : {auc}")
    print(f"  F1 Score: {f1}")
    print(f"\n{report}")

    return {
        "roc_auc": auc, "f1": f1,
        "confusion_matrix": cm, "report": report,
        "y_pred": y_pred, "y_proba": y_proba,
    }


def annual_forest_cover(model, df, feature_cols):
    records = []
    for year in sorted(df["year"].unique()):
        df_year = df[df["year"] == year].copy()
        available = [c for c in feature_cols if c in df_year.columns]
        X = df_year[available].dropna().values
        if len(X) == 0:
            continue
        preds = model.predict(X)
        records.append({
            "year": int(year),
            "forest_cover_pct": round(float(preds.mean() * 100), 2),
            "n_samples": len(preds),
        })
    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_forest_cover_trend(cover_df):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(cover_df["year"], cover_df["forest_cover_pct"],
            marker="o", linewidth=2.5, color="#2d6a4f", markersize=8)
    ax.fill_between(cover_df["year"], cover_df["forest_cover_pct"], alpha=0.15, color="#2d6a4f")
    ax.set_title("Assam Forest Cover Trend", fontsize=14, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Forest Cover (%)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = "outputs/forest_cover_trend.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_feature_importance(importance_df, model_name):
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#2d6a4f" if i < 3 else "#95d5b2" for i in range(len(importance_df))]
    ax.barh(importance_df["feature"][::-1], importance_df["importance"][::-1], color=colors[::-1])
    ax.set_title(f"Feature Importance — {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    path = f"outputs/feature_importance_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_ndvi_distribution(df):
    fig, ax = plt.subplots(figsize=(9, 4))
    for label, color, name in [(1, "#2d6a4f", "Forest"), (0, "#d4a373", "Non-Forest")]:
        subset = df[df["label"] == label]["NDVI"].dropna()
        ax.hist(subset, bins=60, alpha=0.6, color=color, label=name, density=True)
    ax.set_title("NDVI Distribution: Forest vs Non-Forest", fontsize=13, fontweight="bold")
    ax.set_xlabel("NDVI")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    path = "outputs/ndvi_distribution.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_confusion_matrix(cm, model_name):
    # Pad to 2x2 if only one class present
    if cm.shape != (2, 2):
        cm_full = np.zeros((2, 2), dtype=int)
        cm_full[:cm.shape[0], :cm.shape[1]] = cm
        cm = cm_full
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-Forest", "Forest"])
    ax.set_yticklabels(["Non-Forest", "Forest"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}", fontweight="bold")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    path = f"outputs/confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_roc_curves(y_test, results_dict):
    plottable = {k: v for k, v in results_dict.items() if v["roc_auc"] is not None}
    if not plottable:
        print("Skipping ROC curve — no models with valid AUC.")
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = {"Random Forest": "#2d6a4f", "LightGBM": "#52b788"}
    for name, res in plottable.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']})", color=colors.get(name, "blue"), lw=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    path = "outputs/roc_curves.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n🌿 Assam Forest Cover Analysis\n")

    # 1. Data
    print("── Step 1: Data Collection ──────────────────────")
    raw_df = fetch_all_data()

    # 2. Preprocessing
    print("\n── Step 2: Preprocessing ────────────────────────")
    cleaned_df = clean_data(raw_df)
    final_df, feature_cols = add_temporal_features(cleaned_df)
    final_df.to_csv("outputs/processed_data.csv", index=False)
    print("Saved: outputs/processed_data.csv")

    plot_ndvi_distribution(final_df)

    # 3. Train / test split
    print("\n── Step 3: Train/Test Split ─────────────────────")
    X_train, X_test, y_train, y_test, used_features = temporal_split(final_df, feature_cols)

    # 4. Train models
    print("\n── Step 4: Training Models ──────────────────────")
    models_to_train = {
        "Random Forest": build_random_forest(),
        "LightGBM": build_lightgbm(),
    }
    trained_models = {}
    for name, model in models_to_train.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        joblib.dump(model, f"outputs/{name.lower().replace(' ', '_')}.joblib")
        print(f"Saved: outputs/{name.lower().replace(' ', '_')}.joblib")

    # 5. Evaluate
    print("\n── Step 5: Evaluation ───────────────────────────")
    results = {}
    for name, model in trained_models.items():
        results[name] = evaluate_model(model, X_test, y_test, name)
        plot_confusion_matrix(results[name]["confusion_matrix"], name)
        imp = get_feature_importance(model, used_features)
        plot_feature_importance(imp, name)

    plot_roc_curves(y_test, results)

    # Comparison table
    comparison = pd.DataFrame([
        {"Model": name, "ROC-AUC": res["roc_auc"], "F1 Score": res["f1"]}
        for name, res in results.items()
    ])
    print(f"\nModel Comparison:\n{comparison.to_string(index=False)}")
    comparison.to_csv("outputs/model_comparison.csv", index=False)

    # 6. Forest cover trend
    print("\n── Step 6: Annual Forest Cover Trend ────────────")
    best_model_name = comparison.dropna(subset=["ROC-AUC"]).sort_values("ROC-AUC", ascending=False).iloc[0]["Model"]
    print(f"Using best model: {best_model_name}")
    cover_df = annual_forest_cover(trained_models[best_model_name], final_df, used_features)
    cover_df.to_csv("outputs/forest_cover_trend.csv", index=False)
    print(f"Saved: outputs/forest_cover_trend.csv")
    print(f"\n{cover_df.to_string(index=False)}")
    plot_forest_cover_trend(cover_df)

    print("\n✅ All done! Check the outputs/ folder for results.\n")


if __name__ == "__main__":
    main()
