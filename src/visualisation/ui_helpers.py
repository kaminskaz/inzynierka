import streamlit as st
import pandas as pd

def show_csv_preview(df: pd.DataFrame) -> None:
    """Display a preview of the DataFrame in Streamlit."""
    if df.empty:
        st.info("No data to display.")
        return
    st.subheader("Results CSV Preview")
    st.dataframe(df.drop(columns=["filter_id", "ensemble"], errors="ignore"))

def setup_layout() -> None:
    """Inject custom CSS and page title."""
    st.markdown(
        """
        <style>
            .main { max-width: 90% !important; }
            .block-container {
                padding-top: 2rem;
                max-width: 90% !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Streamlit Visualization App")


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