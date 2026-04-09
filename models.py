import joblib
import io
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from config import CONFIG


def build_random_forest():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=10,
            class_weight="balanced",
            n_jobs=-1,
            random_state=CONFIG["seed"],
        )),
    ])


def build_lightgbm():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=10,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            n_jobs=-1,
            random_state=CONFIG["seed"],
            verbose=-1,
        )),
    ])


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def serialize_model(model) -> bytes:
    """Serialize model to bytes for st.download_button."""
    buf = io.BytesIO()
    joblib.dump(model, buf)
    buf.seek(0)
    return buf.read()


def get_feature_importance(model, feature_names):
    import pandas as pd
    clf = model.named_steps["clf"]
    return (
        pd.DataFrame({
            "feature": feature_names,
            "importance": clf.feature_importances_,
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

