# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np

from single_model_plots import *

class StreamlitVisualiser:
    REQUIRED_COLS = [
        "ensemble",
        "version",
        "dataset_name",
        "type_name",
        "model_name",
        "strategy_name",
        "problem_id",
    ]

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = self._load_data()

        if not self.df.empty:
            self._prepare_columns()


    def _load_data(self) -> pd.DataFrame:
        if not os.path.exists(self.csv_path):
            return pd.DataFrame()
        try:
            return pd.read_csv(self.csv_path, dtype={"problem_id": str})
        except Exception as e:
            return pd.DataFrame()

    def _prepare_columns(self):
        """Ensure required columns and derived IDs exist"""

        for col in self.REQUIRED_COLS:
            if col not in self.df.columns:
                self.df[col] = ""

        self.df["ensemble"] = self.df["ensemble"].fillna(False)

        # filter column
        self.df["filter_id"] = np.where(
            self.df["ensemble"],
            # ensemble (multi-model view)
            "Ensemble_ver" +
            self.df["version"].astype(str),
            # single-model view
            self.df["model_name"].astype(str).apply(shorten_model_name) + '_ver' +
            self.df["version"].astype(str)
        )


    @staticmethod
    def _multiselect_filter(df, column, label):
        options = sorted(df[column].dropna().unique())
        selected = st.multiselect(label, options, default=options)
        return df[df[column].isin(selected)] if selected else df

    def select_model(self):
        options = sorted(self.df["filter_id"].unique())
        return st.selectbox("Select Model", options)

    @staticmethod
    def show_csv_preview(df):
        if df.empty:
            st.info("No data to display.")
            return

        st.subheader("Results CSV Preview")
        st.dataframe(
            df.drop(columns=["filter_id", "ensemble"], errors="ignore")
        )

    def run(self):
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

        if self.df.empty:
            st.info("No CSV loaded. Please provide a valid CSV file.")
            return

        vis_type = st.radio(
            "Select Visualization Type",
            ["single", "multiple"],
            horizontal=True,
        )

        # ==============================================================
        # SINGLE MODEL VIEW
        # ==============================================================
        st.subheader("Model Selection")
        if vis_type == "single":
             
            single_model_id = self.select_model()
            df_model = self.df[self.df["filter_id"] == single_model_id].copy()

            if df_model.empty:
                st.info("No data for selected model.")
                return

            df_single = df_model.copy()

            df_single["dataset_name"] = df_single["dataset_name"].fillna("Unknown Dataset")
            df_single = self._multiselect_filter(
                df_single, "dataset_name", "Select Dataset(s)"
            )

            is_ensemble = bool(df_single["ensemble"].iloc[0])

            strategy_column_name = map_strategy_column_name(is_ensemble)
            df_single = self._multiselect_filter(
                    df_single, strategy_column_name, "Select Type(s)" if is_ensemble else "Select Strategy(ies)")

            if df_single.empty:
                st.info("No data matches the selected filters.")
                return

            self.show_csv_preview(df_single)
            plot_judged_answers(df_single, strategy_col=strategy_column_name)
            plot_confidence(df_single, strategy_col=strategy_column_name)

            st.subheader("Details for Chosen Dataset")

            selected_dataset = st.selectbox(
                "Select Dataset",
                sorted(df_single["dataset_name"].unique()),
            )

            df_dataset = df_model[df_model["dataset_name"] == selected_dataset]

            show_problem_strategy_table(df_dataset, dataset_name=selected_dataset, strategy_col=strategy_column_name)

            if is_ensemble:
                st.subheader("Evaluation Summary for Chosen Dataset and Type")
            else:
                st.subheader("Evaluation Summary for Chosen Dataset and Strategy")

            selected_strategy = st.selectbox(
                "Select Strategy" if not is_ensemble else "Select Type",
                sorted(df_dataset[strategy_column_name].unique()),
            )

            display_evaluation_summary(df_dataset, dataset_name=selected_dataset, strategy_name=selected_strategy, strategy_col=strategy_column_name, is_ensemble=is_ensemble)
            
            st.divider()
            st.subheader("Sample Problem Details")
            
            selected_problem_id = st.selectbox(
                "Select Problem ID",
                sorted(df_dataset[df_dataset[strategy_column_name] == selected_strategy]["problem_id"].unique()),
            )

            show_chosen_problem(
                df_single,
                problem_id=selected_problem_id,
                dataset_name=selected_dataset,
                strategy_name=selected_strategy,
                strategy_col=strategy_column_name
            )
            
            if not is_ensemble:
                show_single_model_config(
                    dataset_name=selected_dataset,
                    model_name=df_single["model_name"].iloc[0],
                    strategy_name=selected_strategy,
                    version=df_single["version"].iloc[0]               
                )
                
            else:
                show_ensemble_config(
                    dataset_name=selected_dataset,
                    type_name=selected_strategy,
                    ensemble_version=df_single["version"].iloc[0]               
                )

        # ==============================================================
        # MULTI MODEL VIEW
        # ==============================================================

        else:
            selected_filters = self.select_multiple_models()
            if not selected_filters:
                st.info("Select at least one model or ensemble.")
                return

            df_filtered = self.df[self.df["filter_id"].isin(selected_filters)].copy()
            self.show_csv_preview(df_filtered)


if __name__ == "__main__":
    visualiser = StreamlitVisualiser("results/all_results_concat.csv")
    visualiser.run()
