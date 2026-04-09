# app.py
import streamlit as st
import pandas as pd
import numpy as np
import json

from config import CONFIG
from data_collection import initialize_gee, get_assam_boundary, fetch_samples_for_year
from preprocessing import clean_data, add_temporal_features, temporal_split
from models import (
    build_random_forest, build_lightgbm,
    train_model, serialize_model, get_feature_importance,
)
from evaluate import evaluate_model, annual_forest_cover
from visualize import (
    forest_cover_trend_fig,
    feature_importance_fig,
    ndvi_distribution_fig,
    confusion_matrix_fig,
    roc_curve_fig,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Assam Forest Cover Analysis",
    page_icon="🌿",
    layout="wide",
)

# ── Session state defaults ─────────────────────────────────────────────────────
for key in ["gee_ready", "df", "models", "results", "cover_df", "features"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌿 Assam Forest Monitor")
    st.markdown("---")

    st.subheader("⚙️ Configuration")
    start_year = st.slider("Start Year", 2015, 2022, CONFIG["start_year"])
    end_year = st.slider("End Year", 2016, 2023, CONFIG["end_year"])
    forest_threshold = st.slider("Forest Cover Threshold (%)", 10, 50, CONFIG["forest_threshold"])
    cloud_threshold = st.slider("Cloud Cover Threshold (%)", 10, 60, CONFIG["cloud_threshold"])
    samples_per_class = st.slider("Samples per Class per Year", 500, 3000, 1500, step=500)
    test_years_input = st.multiselect(
        "Test Years",
        options=list(range(start_year, end_year + 1)),
        default=[end_year - 1, end_year],
    )

    # Update config from UI
    CONFIG.update({
        "start_year": start_year,
        "end_year": end_year,
        "forest_threshold": forest_threshold,
        "cloud_threshold": cloud_threshold,
    })

    st.markdown("---")
    st.subheader("🔑 GEE Authentication")
    st.markdown(
        "Paste your GEE service account JSON key below.\n\n"
        "[How to get a service account key →](https://developers.google.com/earth-engine/guides/service_account)"
    )
    gee_key_input = st.text_area("Service Account JSON Key", height=120, placeholder='{"type": "service_account", ...}')

    if st.button("🔌 Connect to GEE", use_container_width=True):
        if not gee_key_input.strip():
            st.error("Please paste your GEE service account JSON key.")
        else:
            try:
                key_dict = json.loads(gee_key_input)
                initialize_gee(key_dict)
                st.session_state.gee_ready = True
                st.success("Connected to GEE!")
            except Exception as e:
                st.error(f"Connection failed: {e}")

    st.markdown("---")
    st.caption("Built with Google Earth Engine + scikit-learn + LightGBM")

# ── Main content ───────────────────────────────────────────────────────────────
st.title("🌳 Assam Forest Cover Analysis")
st.markdown("Analyze forest cover change in Assam (2015–2023) using Landsat, ERA5 climate data, and ML.")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📡 Data Collection",
    "🤖 Model Training",
    "📊 Results & Charts",
    "🗺️ Forest Cover Trend",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Data Collection
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Data Collection from Google Earth Engine")

    if not st.session_state.gee_ready:
        st.warning("Connect to GEE using the sidebar before fetching data.")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(
                "Fetches annual Landsat composites, ERA5 climate data, "
                "Hansen forest labels, and SRTM topography for Assam — "
                "directly into memory (no Drive export needed)."
            )
        with col2:
            fetch_btn = st.button("🛰️ Fetch Data from GEE", use_container_width=True)

        if fetch_btn:
            assam = get_assam_boundary()
            all_dfs = []

            progress = st.progress(0)
            status = st.empty()
            years = list(range(CONFIG["start_year"], CONFIG["end_year"] + 1))

            for i, year in enumerate(years):
                status.text(f"Fetching {year}...")
                try:
                    df_year = fetch_samples_for_year(assam, year)
                    all_dfs.append(df_year)
                except Exception as e:
                    st.warning(f"Could not fetch {year}: {e}")
                progress.progress((i + 1) / len(years))

            if all_dfs:
                raw_df = pd.concat(all_dfs, ignore_index=True)
                cleaned_df, dropped = clean_data(raw_df)
                final_df, feature_cols = add_temporal_features(cleaned_df)

                st.session_state.df = final_df
                st.session_state.features = feature_cols

                status.text("Done!")
                progress.empty()
                st.success(f"Fetched {len(final_df):,} samples across {len(all_dfs)} years. Dropped {dropped:,} invalid rows.")
            else:
                st.error("No data fetched. Check your GEE connection and permissions.")

        # Show data preview if available
        if st.session_state.df is not None:
            df = st.session_state.df
            st.subheader("Data Preview")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Samples", f"{len(df):,}")
            col2.metric("Years", df["year"].nunique())
            col3.metric("Forest Samples", f"{(df['label'] == 1).sum():,}")
            col4.metric("Non-Forest Samples", f"{(df['label'] == 0).sum():,}")

            st.dataframe(df.head(100), use_container_width=True)

            st.subheader("NDVI Distribution")
            st.pyplot(ndvi_distribution_fig(df))

            # Download raw data
            csv = df.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download Processed Data as CSV",
                data=csv,
                file_name="assam_forest_data.csv",
                mime="text/csv",
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Training
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Model Training")

    # Allow uploading a previously downloaded CSV instead of re-fetching
    st.markdown("**Option A:** Use data fetched in Tab 1 &nbsp;|&nbsp; **Option B:** Upload a previously saved CSV")
    uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])
    if uploaded:
        df_upload = pd.read_csv(uploaded)
        _, feature_cols = add_temporal_features(df_upload)
        st.session_state.df = df_upload
        st.session_state.features = feature_cols
        st.success(f"Loaded {len(df_upload):,} rows from uploaded file.")

    if st.session_state.df is None:
        st.info("Fetch or upload data first.")
    else:
        df = st.session_state.df
        features = st.session_state.features

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Training Settings")
            test_years = test_years_input if test_years_input else [CONFIG["end_year"]]
            st.write(f"Train years: {[y for y in df['year'].unique() if y not in test_years]}")
            st.write(f"Test years:  {test_years}")

            models_to_train = st.multiselect(
                "Models to train",
                ["Random Forest", "LightGBM"],
                default=["Random Forest", "LightGBM"],
            )

        with col2:
            st.subheader("Feature Selection")
            selected_features = st.multiselect(
                "Select features",
                options=features,
                default=features,
            )

        train_btn = st.button("🚀 Train Models", use_container_width=True)

        if train_btn:
            if not models_to_train:
                st.error("Select at least one model.")
            elif not selected_features:
                st.error("Select at least one feature.")
            else:
                X_train, X_test, y_train, y_test, used_features = temporal_split(
                    df, test_years, selected_features
                )

                trained = {}
                results = {}
                progress = st.progress(0)

                for i, model_name in enumerate(models_to_train):
                    with st.spinner(f"Training {model_name}..."):
                        model = (
                            build_random_forest() if model_name == "Random Forest"
                            else build_lightgbm()
                        )
                        model = train_model(model, X_train, y_train)
                        trained[model_name] = model
                        results[model_name] = evaluate_model(model, X_test, y_test)
                        results[model_name]["y_test"] = y_test
                    progress.progress((i + 1) / len(models_to_train))

                st.session_state.models = trained
                st.session_state.results = results
                st.session_state.features = used_features
                progress.empty()
                st.success("Training complete! Go to Results tab.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Results & Charts
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Results & Model Evaluation")

    if st.session_state.results is None:
        st.info("Train models first.")
    else:
        results = st.session_state.results
        models = st.session_state.models
        features = st.session_state.features

        # Metrics comparison table
        st.subheader("Model Comparison")
        comparison = pd.DataFrame([
            {
                "Model": name,
                "ROC-AUC": res["roc_auc"],
                "F1 Score": res["f1"],
                "Forest Precision": round(res["report"]["Forest"]["precision"], 4),
                "Forest Recall": round(res["report"]["Forest"]["recall"], 4),
            }
            for name, res in results.items()
        ])
        st.dataframe(comparison, use_container_width=True)

        # Per-model details
        for model_name, res in results.items():
            with st.expander(f"📋 {model_name} — Detailed Results", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("ROC-AUC", res["roc_auc"])
                col2.metric("F1 Score", res["f1"])
                col3.metric("Forest Precision", round(res["report"]["Forest"]["precision"], 3))
                col4.metric("Forest Recall", round(res["report"]["Forest"]["recall"], 3))

                c1, c2 = st.columns(2)
                with c1:
                    st.pyplot(confusion_matrix_fig(res["confusion_matrix"], model_name))
                with c2:
                    imp = get_feature_importance(models[model_name], features)
                    st.pyplot(feature_importance_fig(imp, model_name))

        # ROC curves
        if len(results) > 1:
            st.subheader("ROC Curve Comparison")
            y_test = list(results.values())[0]["y_test"]
            st.pyplot(roc_curve_fig(y_test, results))

        # Model downloads
        st.subheader("Download Trained Models")
        cols = st.columns(len(models))
        for col, (name, model) in zip(cols, models.items()):
            col.download_button(
                f"⬇️ Download {name}",
                data=serialize_model(model),
                file_name=f"{name.lower().replace(' ', '_')}.joblib",
                mime="application/octet-stream",
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Forest Cover Trend
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Annual Forest Cover Trend")

    if st.session_state.models is None or st.session_state.df is None:
        st.info("Train models first.")
    else:
        models = st.session_state.models
        df = st.session_state.df
        features = st.session_state.features

        selected_model = st.selectbox("Select model for trend analysis", list(models.keys()))

        if st.button("📈 Compute Annual Forest Cover"):
            with st.spinner("Predicting across all years..."):
                cover_df = annual_forest_cover(models[selected_model], df, features)
                st.session_state.cover_df = cover_df

        if st.session_state.cover_df is not None:
            cover_df = st.session_state.cover_df

            # Summary metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Peak Forest Cover", f"{cover_df['forest_cover_pct'].max():.1f}%",
                        f"({int(cover_df.loc[cover_df['forest_cover_pct'].idxmax(), 'year'])})")
            col2.metric("Lowest Forest Cover", f"{cover_df['forest_cover_pct'].min():.1f}%",
                        f"({int(cover_df.loc[cover_df['forest_cover_pct'].idxmin(), 'year'])})")
            total_change = cover_df['forest_cover_pct'].iloc[-1] - cover_df['forest_cover_pct'].iloc[0]
            col3.metric("Total Change", f"{total_change:+.1f}%",
                        delta_color="inverse")

            st.pyplot(forest_cover_trend_fig(cover_df))
            st.dataframe(cover_df, use_container_width=True)

            csv = cover_df.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download Forest Cover Trend CSV",
                data=csv,
                file_name="assam_forest_cover_trend.csv",
                mime="text/csv",
            )

