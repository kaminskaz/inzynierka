import os
import pandas as pd
import numpy as np
from src.visualisation.ui_helpers import shorten_model_name
import streamlit as st
import json

REQUIRED_COLS = [
    "ensemble",
    "version",
    "dataset_name",
    "type_name",
    "model_name",
    "strategy_name",
    "problem_id",
]

SCORE_COLOR_MAP = {
    "Right": "#95cd59",
    "Somewhat right": "#f3be20",
    "Somewhat wrong": "#f38320",
    "Wrong": "#d62728",
    "Unclear": "#7f7f7f",
    "No answer provided": "#69aee3"
}

def load_results(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        return pd.DataFrame()

    df = pd.read_csv(csv_path, dtype={"problem_id": str})
    return prepare_columns(df)

def prepare_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = ""

    df["ensemble"] = df["ensemble"].fillna(False)

    df["filter_id"] = np.where(
        df["ensemble"],
        "Ensemble_ver" + df["version"].astype(str),
        df["model_name"].astype(str).apply(shorten_model_name)
        + "_ver" + df["version"].astype(str),
    )

    df["judge_filter_id"] = np.where(
        df["judge_model_name"].notna(),
        df["judge_model_name"].astype(str)
        + np.where(
            df["judge_model_param_set"].notna(),
            "_" + df["judge_model_param_set"].astype("Int64").astype(str),
            ""
        ),
        ""
    )

    return df

def load_json_safe(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Failed to load JSON: {e}")
        return None

def validate_columns(df, required_cols):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.warning(f"Missing required columns: {missing}")
        return False
    return True

def get_present_colors(df, score_col="score"):
    present_labels = df[score_col].dropna().unique().tolist()
    return {label: SCORE_COLOR_MAP[label] for label in present_labels if label in SCORE_COLOR_MAP}

