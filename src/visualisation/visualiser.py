# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from single_model_plots import *

class StreamlitVisualiser:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None
        self.load_data()

        if self.df is not None and not self.df.empty:
            self.create_filter_column()
            self.create_single_model_column()

    def load_data(self):
        try:
            self.df = pd.read_csv(self.csv_path, dtype={'problem_id': str})
        except Exception as e:
            st.error(f"Failed to load CSV: {e}")
            self.df = pd.DataFrame()

    # ------------------------------------------------------------------
    # FILTER COLUMNS
    # ------------------------------------------------------------------

    def create_filter_column(self):
        """Row-wise filter_id (ensemble OR model strategy)"""

        required_cols = [
            'ensemble', 'version', 'dataset_name', 'type_name',
            'model_name', 'strategy_name'
        ]

        for col in required_cols:
            if col not in self.df.columns:
                self.df[col] = ""

        self.df['filter_id'] = np.where(
            self.df['ensemble'].notna() & (self.df['ensemble']),
            "Ensemble" + "_" +
            self.df['version'].astype(str) + "_" +
            self.df['dataset_name'].astype(str) + "_" +
            self.df['type_name'].astype(str),
            self.df['model_name'].astype(str) + "_" +
            self.df['version'].astype(str) + "_" +
            self.df['dataset_name'].astype(str) + "_" +
            self.df['strategy_name'].astype(str)
        )

    def create_single_model_column(self):
        """Single-model selector: ensemble OR model_name + version"""

        self.df['single_model_id'] = np.where(
            self.df['ensemble'].notna() & (self.df['ensemble']),
            self.df['ensemble'].astype(str) + "_" +
            self.df['version'].astype(str),
            self.df['model_name'].astype(str) + "_" +
            self.df['version'].astype(str),
        )

    def select_datasets_for_single(self, df_single_model):
        """Dataset selector limited to the chosen single model"""
        if 'dataset_name' not in df_single_model.columns:
            return None

        datasets = sorted(df_single_model['dataset_name'].unique())
        return st.multiselect(
            "Select Dataset(s)",
            options=datasets,
            default=datasets  # sensible default: all
        )
    
    def select_strategies_for_single(self, df_single_model):
        """Strategy selector limited to the chosen single model"""
        if 'strategy_name' not in df_single_model.columns:
            return None

        strategies = sorted(df_single_model['strategy_name'].unique())
        return st.multiselect(
            "Select Strategy(s)",
            options=strategies,
            default=strategies  # sensible default: all
        )
        

    # ------------------------------------------------------------------
    # FILTER UI
    # ------------------------------------------------------------------

    def select_single_model(self):
        options = sorted(self.df['single_model_id'].unique())
        return st.selectbox(
            "Select Model",
            options=options
        )
    
    def filter_single_model_with_dataset(self, single_model_id, datasets):
        df = self.df[self.df['single_model_id'] == single_model_id]

        if datasets:
            df = df[df['dataset_name'].isin(datasets)]

        return df

    def filter_single_model(self, single_model_id):
        return self.df[self.df['single_model_id'] == single_model_id]

    def select_multiple_models(self):
        options = sorted(self.df['filter_id'].unique())
        return st.multiselect("Select Models / Ensembles", options=options)

    def filter_multiple_models(self, selected_filters):
        if selected_filters:
            return self.df[self.df['filter_id'].isin(selected_filters)]
        return self.df.copy()

    # ------------------------------------------------------------------
    # DISPLAY
    # ------------------------------------------------------------------

    def show_csv_preview(self, df_filtered):
        if df_filtered is not None and not df_filtered.empty:
            st.subheader("CSV Preview")
            df_display = df_filtered.drop(
                columns=['filter_id', 'single_model_id'],
                errors='ignore'
            )
            st.dataframe(df_display.head(10))
        else:
            st.info("No data to display.")


    # ------------------------------------------------------------------
    # MAIN APP
    # ------------------------------------------------------------------

    def run(self):
        st.markdown(
            """
            <style>
                .main { max-width: 90% !important; }
                .block-container {
                    padding-top: 2rem;
                    padding-left: 10;
                    padding-right: 10;
                    max-width: 90% !important;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.title("Streamlit Visualization App")

        # visualization type
        vis_type = st.radio(
            "Select Visualization Type",
            options=["single", "multiple"],
            horizontal=True
        )

        if vis_type == "single":
            st.subheader("Single Model Selection")
            single_model = self.select_single_model()
            df_single = self.df[self.df['single_model_id'] == single_model]

            # Dataset selector for the single model
            df_single['dataset_name'] = df_single['dataset_name'].fillna("Unknown Dataset")
            selected_datasets = st.multiselect(
                "Select Dataset(s)",
                options=sorted(df_single['dataset_name'].unique()),
                default=sorted(df_single['dataset_name'].unique())
            )
            df_filtered = df_single[df_single['dataset_name'].isin(selected_datasets)]

            if df_filtered.empty:
                st.info("No data matches the selected filters.")
                return

            is_ensemble = df_filtered['ensemble'].iloc[0]

            if is_ensemble:
                df_filtered['type_name'] = df_filtered['type_name'].fillna("Unknown Type")

                # Type selector
                type_options = sorted(df_filtered['type_name'].unique())
                selected_types = st.multiselect(
                    "Select Type(s)",
                    options=type_options,
                    default=type_options
                )
                df_filtered = df_filtered[df_filtered['type_name'].isin(selected_types)]

            else:

                # Strategy selector
                strategy_options = sorted(df_filtered['strategy_name'].unique())
                selected_strategies = st.multiselect(
                    "Select Strategy(s)",
                    options=strategy_options,
                    default=strategy_options
                )
                df_filtered = df_filtered[df_filtered['strategy_name'].isin(selected_strategies)]

            self.show_csv_preview(df_filtered)
            plot_judged_answers(df_filtered)

            st.subheader(f"Details for chosen Dataset")

            selected_dataset = st.selectbox(
                "Select Dataset for Problem Details:",
                options=sorted(df_filtered['dataset_name'].unique())
            )

            df_dataset = df_filtered[df_filtered['dataset_name'] == selected_dataset]

            selected_problem_id = st.selectbox(
                "Select Problem ID for Details:",
                options=sorted(df_dataset['problem_id'].unique())
            )

            show_chosen_problem(
                df_dataset,
                problem_id=selected_problem_id,
                dataset_name=selected_dataset
            )

            show_problem_strategy_table(
                df_filtered,
                dataset_name=selected_dataset
            )



        else:
            st.subheader("Multiple Models Selection")
            selected_filters = self.select_multiple_models()
            df_filtered = self.filter_multiple_models(selected_filters)



if __name__ == "__main__":
    app = StreamlitVisualiser("results/all_results_concat.csv")
    app.run()
