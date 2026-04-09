import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "report": classification_report(
            y_test, y_pred,
            target_names=["Non-Forest", "Forest"],
            output_dict=True,
        ),
        "y_pred": y_pred,
        "y_proba": y_proba,
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

