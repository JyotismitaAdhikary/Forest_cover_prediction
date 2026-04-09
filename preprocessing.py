import pandas as pd
import numpy as np
import streamlit as st

FEATURE_COLS = ["NDVI", "EVI", "mean_temp", "total_precip", "elevation", "slope"]
TARGET_COL = "label"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    initial = len(df)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    df = df[df["NDVI"].between(-1, 1)]
    df = df[df["EVI"].between(-1, 1)]
    df = df[df["elevation"] >= 0]
    dropped = initial - len(df)
    return df.reset_index(drop=True), dropped


def add_temporal_features(df: pd.DataFrame):
    df = df.sort_values(["longitude", "latitude", "year"]).reset_index(drop=True)
    grp = ["longitude", "latitude"]
    df["NDVI_lag1"] = df.groupby(grp)["NDVI"].shift(1)
    df["NDVI_roll3"] = (
        df.groupby(grp)["NDVI"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )
    df["NDVI_delta"] = df["NDVI"] - df["NDVI_lag1"]

    extended = FEATURE_COLS + ["NDVI_lag1", "NDVI_roll3", "NDVI_delta"]
    return df, extended


def temporal_split(df: pd.DataFrame, test_years: list, feature_cols: list):
    train_df = df[~df["year"].isin(test_years)]
    test_df = df[df["year"].isin(test_years)]

    available = [c for c in feature_cols if c in df.columns]
    X_train = train_df[available].values
    y_train = train_df[TARGET_COL].astype(int).values
    X_test = test_df[available].values
    y_test = test_df[TARGET_COL].astype(int).values

    return X_train, X_test, y_train, y_test, available

