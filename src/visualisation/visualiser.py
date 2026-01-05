import streamlit as st
import pandas as pd
import argparse
import os

from src.visualisation.data_handling import load_results, prepare_columns
from src.visualisation.ui_helpers import multiselect_filter, map_strategy_column_name
from src.visualisation.display import *

class StreamlitVisualiser:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = load_results(csv_path)
        if not self.df.empty:
            self.df = prepare_columns(self.df)

    def run(self):
        setup_layout()

        if self.df.empty:
            st.info("No CSV loaded. Please provide a valid CSV file.")
            return

        df_model = self._select_model()
        if df_model is None:
            return

        df_single, is_ensemble, strategy_col = self._apply_filters(df_model)
        if df_single is None:
            return

        self._show_overview(df_single, strategy_col)
        self._show_dataset_section(df_model, is_ensemble, strategy_col)


    def _select_model(self) -> pd.DataFrame | None:
        options = sorted(self.df["filter_id"].unique())
        selected_id = st.selectbox("Select Model", options)
        df_model = self.df[self.df["filter_id"] == selected_id]
        if df_model.empty:
            st.info("No data for selected model.")
            return None
        return df_model

    def _apply_filters(self, df_model: pd.DataFrame) -> tuple[pd.DataFrame | None, bool | None, str | None]:
        df = df_model.copy()
        df["dataset_name"] = df["dataset_name"].fillna("Unknown Dataset")
        df = multiselect_filter(df, "dataset_name", "Select Dataset(s)")
        if df.empty:
            return None, None, None

        is_ensemble = bool(df["ensemble"].iloc[0])
        strategy_col = map_strategy_column_name(is_ensemble)
        df = multiselect_filter(
            df, strategy_col, "Select Type(s)" if is_ensemble else "Select Strategy(ies)"
        )
        if df.empty:
            st.info("No data matches the selected filters.")
            return None, None, None

        return df, is_ensemble, strategy_col

    def _show_overview(self, df_single: pd.DataFrame, strategy_col: str) -> None:
        show_csv_preview(df_single)
        plot_judged_answers(df_single, strategy_col=strategy_col)
        plot_confidence(df_single, strategy_col=strategy_col)

    def _show_dataset_section(self, df_model: pd.DataFrame, is_ensemble: bool, strategy_col: str) -> None:
        st.subheader("Details for Chosen Dataset")
        selected_dataset = st.selectbox("Select Dataset", sorted(df_model["dataset_name"].unique()))
        df_dataset = df_model[df_model["dataset_name"] == selected_dataset]

        show_problem_strategy_table(df_dataset, dataset_name=selected_dataset, strategy_col=strategy_col)

        st.subheader("Evaluation Summary for Chosen Dataset and " + ("Type" if is_ensemble else "Strategy"))
        selected_strategy = st.selectbox(
            "Select Strategy" if not is_ensemble else "Select Type",
            sorted(df_dataset[strategy_col].unique())
        )
        display_evaluation_summary(
            df_dataset,
            dataset_name=selected_dataset,
            strategy_name=selected_strategy,
            strategy_col=strategy_col,
            is_ensemble=is_ensemble
        )

        st.divider()
        st.subheader("Sample Problem Details")
        selected_problem_id = st.selectbox(
            "Select Problem ID",
            sorted(df_dataset[df_dataset[strategy_col] == selected_strategy]["problem_id"].unique())
        )
        show_chosen_problem(
            df_dataset,
            problem_id=selected_problem_id,
            dataset_name=selected_dataset,
            strategy_name=selected_strategy,
            strategy_col=strategy_col
        )

        if not is_ensemble:
            show_single_model_config(
                dataset_name=selected_dataset,
                model_name=df_dataset["model_name"].iloc[0],
                strategy_name=selected_strategy,
                version=df_dataset["version"].iloc[0]
            )
        else:
            show_ensemble_config(
                dataset_name=selected_dataset,
                type_name=selected_strategy,
                ensemble_version=df_dataset["version"].iloc[0]
            )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", nargs="?", default=os.path.join("results", "all_results_concat.csv"))
    args = parser.parse_args()
    print("Using CSV path:", args.csv_path)

    visualiser = StreamlitVisualiser(csv_path=args.csv_path)
    visualiser.run()

if __name__ == "__main__":
    main()