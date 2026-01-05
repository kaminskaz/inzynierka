import streamlit as st
import pandas as pd
import os
import json

def multiselect_filter(df: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    options = sorted(df[column].dropna().unique())
    selected = st.multiselect(label, options, default=options)
    return df[df[column].isin(selected)] if selected else df

def safe_display(value):
    if pd.isna(value) or value == "":
        return "-"
    return value

def map_strategy_column_name(is_ensemble):
    if is_ensemble:
        return "type_name"
    else:
        return "strategy_name"
    
def shorten_model_name(model_name: str) -> str:
    parts = model_name.split('/')
    if len(parts) >= 3:
        short_model_name = parts[1]
    elif len(parts) == 2:
        short_model_name = parts[1]
    else:
        short_model_name = model_name
    short_model_name = short_model_name.replace('/', '_')
    return short_model_name
    
def render_markdown(label: str, value, font_size=18):
    st.markdown(
        f"<p style='font-size:{font_size}px;'>"
        f"<b>{safe_display(label)}</b>: {safe_display(value)}"
        f"</p>",
        unsafe_allow_html=True
    )