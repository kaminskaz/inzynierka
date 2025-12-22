import os
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import json
import seaborn as sns

def _safe_display(value):
    if pd.isna(value) or value == "":
        return "N/A"
    return value


def plot_judged_answers(df, score_col="score", dataset_col="dataset_name", strategy_col="strategy_name"):
    st.subheader("Scored Answers")

    if df.empty:
        st.info("No valid data to display.")
        return

    for col in [score_col, dataset_col, strategy_col]:
        if col not in df.columns:
            st.warning(f"Column '{col}' not found.")
            return
        if df[col].isnull().all():
            st.warning(f"Column '{col}' contains only null values.")
            return

    # remove invalid scores ex. LLM evaluation failed
    # df = df[~df[score_col].isin(["", "LLM evaluation failed", None])]

    datasets = df[dataset_col].dropna().unique()
    ncols = 2
    nrows = (len(datasets) + 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows))

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    for i, dataset in enumerate(datasets):
        ax = axes[i]
        df_ds = df[df[dataset_col] == dataset]

        grouped = df_ds.groupby([strategy_col, score_col]).size().reset_index(name='count')
        pivot = grouped.pivot(index=strategy_col, columns=score_col, values='count').fillna(0)

        pivot.plot(kind='bar', stacked=False, ax=ax)
        ax.set_title(f"{dataset}")
        ax.set_xlabel("Strategy")
        ax.set_ylabel("Count")
        ax.legend(title="Score", bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    st.pyplot(fig)


def show_chosen_problem(df, problem_id, dataset_name):
    st.subheader(f"Details for Problem ID: {problem_id}")
    
    if df.empty:
        st.info("No valid data to display.")
        return

    row = df[(df['problem_id'] == problem_id) & (df['dataset_name'] == dataset_name)].iloc[0]

    col1, col2 = st.columns([1, 1])

    with col1:
        if dataset_name is not None:
            img_path = os.path.join("data", dataset_name, "problems", str(problem_id), "question_panel.png")

            if os.path.exists(img_path):
                st.image(str(img_path), caption=f"{dataset_name} – Problem {problem_id}")
            else:
                st.warning(f"Image not found:\n{img_path}")
        else:
            st.warning("Dataset name not available.")

    with col2:
        st.markdown("### Model Answer")
        st.write(_safe_display(row.get('answer')))

        st.markdown("### Confidence")
        st.write(_safe_display(row.get('confidence')))

        st.markdown("### Key answer")
        st.write(_safe_display(row.get('key')))

        st.markdown("### Score")
        st.write(_safe_display(row.get('score')))

        st.markdown("### Rationale")
        st.write(_safe_display(row.get('rationale')))

def show_model_config(
        model_name, 
        dataset_name, 
        strategy_name, 
        version, 
        type_name, 
        ensemble
    ):
    
    is_ensemble = False
    if ensemble.lower() == 'ensemble':
        is_ensemble = True

    st.subheader("Model Configuration")

    if is_ensemble:
        components = ["results", "ensembles", dataset_name, type_name, f"ensemble_ver{version}", "ensemble_config.json"]
    else:
        components = ["results", dataset_name, strategy_name, model_name, f"ver{version}", "metadata.json"]

    if any(c is None or (isinstance(c, float) and pd.isna(c)) or str(c).strip() == "" for c in components):
        st.warning("Cannot build config path: one or more required fields are missing or invalid.")
        return

    metadata_path = os.path.join(*[str(c) for c in components])

    if not os.path.exists(metadata_path):
        st.warning(f"Configuration file not found:\n`{metadata_path}`")
        return

    try:
        if isinstance(metadata_path, str):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        st.json(metadata)
    except Exception:
        st.write(metadata)


def show_problem_strategy_table(
    df,
    dataset_name,
    problem_col="problem_id",
    strategy_col="strategy_name",
    outcome_col="score"
):
    st.subheader("Problem × Strategy Outcome Overview")

    df = df[df['dataset_name'] == dataset_name].copy()
    if df.empty:
        st.info("No data for selected dataset.")
        return

    outcome_map = {
        "Right": 1,
        "Wrong": 0,
    }

    df["outcome_val"] = df[outcome_col].map(outcome_map)
    df = df.dropna(subset=["outcome_val"])

    if df.empty:
        st.info("No valid outcomes to display.")
        return

    pivot = df.pivot(
        index=problem_col,
        columns=strategy_col,
        values="outcome_val"
    ).sort_index()

    def color_cells(val):
        if val == 1:
            return "background-color: #1a9850; color: white"   
        if val == 0:
            return "background-color: #d73027; color: white" 
        return "background-color: #f0f0f0"                 

    styled = (
        pivot
        .style
        .applymap(color_cells)
    )

    st.dataframe(
        styled,
        height=450,                 
        use_container_width=True
    )
