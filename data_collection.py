import ee
import streamlit as st
from config import CONFIG


def initialize_gee(service_account_key: dict):
    """
    Initialize GEE using a service account JSON key.
    On Streamlit Cloud, store the key in st.secrets.
    """
    credentials = ee.ServiceAccountCredentials(
        email=service_account_key["client_email"],
        key_data=service_account_key["private_key"],
    )
    ee.Initialize(credentials)


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
    collection = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(region)
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .filter(ee.Filter.lt("CLOUD_COVER", CONFIG["cloud_threshold"]))
        .map(mask_landsat_clouds)
        .map(compute_indices)
    )
    return collection.median().select(["NDVI", "EVI"]).clip(region)


def get_hansen_labels(region):
    hansen = ee.Image("UMD/hansen/global_forest_change_2023_v1_11")
    tree_cover = hansen.select("treecover2000")
    loss_year = hansen.select("lossyear")
    end_yr_offset = CONFIG["end_year"] - 2000

    forest_mask = tree_cover.gte(CONFIG["forest_threshold"]).And(
        loss_year.eq(0).Or(loss_year.gt(end_yr_offset))
    )
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


@st.cache_data(show_spinner=False)
def fetch_samples_for_year(_region, year):
    """
    Fetch pixel samples for one year directly into a DataFrame.
    Cached so re-runs don't re-query GEE.
    Underscore prefix on _region tells Streamlit not to hash the EE object.
    """
    label, loss_band = get_hansen_labels(_region)
    topo = get_topographic_data(_region)
    composite = get_annual_composite(year, _region)
    climate = get_climate_data(year, _region)

    stacked = (
        composite
        .addBands(climate)
        .addBands(topo)
        .addBands(label)
        .addBands(loss_band)
    )

    samples = stacked.stratifiedSample(
        numPoints=1500,
        classBand="label",
        region=_region.geometry(),
        scale=CONFIG["scale"],
        seed=CONFIG["seed"],
        geometries=True,
    )

    # Convert to DataFrame directly — no Drive export needed
    features = samples.getInfo()
    rows = [f["properties"] for f in features["features"]]
    import pandas as pd
    df = pd.DataFrame(rows)
    df["year"] = year
    return df

