import os
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import json
import seaborn as sns

# from src.technical.utils import shorten_model_name

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


def _safe_display(value):
    if pd.isna(value) or value == "":
        return "-"
    return value

def map_strategy_column_name(is_ensemble):
    if is_ensemble:
        return "type_name"
    else:
        return "strategy_name"


def plot_judged_answers(df, score_col="score", dataset_col="dataset_name", strategy_col="strategy_name"):

    st.subheader("Scored Answers Summary")

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

    datasets = df[dataset_col].dropna().unique()
    if len(datasets) == 0:
        st.info("No datasets with valid data.")
        return

    label_color_map = {
        "Right": "#95cd59",
        "Somewhat right": "#f3be20",
        "Somewhat wrong": "#f38320",
        "Wrong": "#d62728",
        "Unclear": "#7f7f7f",
        "No answer provided": "#69aee3"
    }

    present_labels = df[score_col].dropna().unique().tolist()
    present_colors = {label: label_color_map[label] for label in present_labels if label in label_color_map}

    ncols = 2
    nrows = (len(datasets) + 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*6, nrows*5))
    axes = np.array(axes).flatten() if isinstance(axes, np.ndarray) else np.array([axes])

    max_count = df.groupby([dataset_col, strategy_col, score_col]).size().max()

    for i, dataset in enumerate(datasets):
        df_ds = df[df[dataset_col] == dataset]

        if df_ds.empty or df_ds[strategy_col].dropna().empty:
            axes[i].set_visible(False)
            continue

        ax = axes[i]
        sns.countplot(
            data=df_ds,
            x=strategy_col,
            hue=score_col,
            palette=present_colors,
            ax=ax,
            order=sorted(df_ds[strategy_col].unique()),
            width=0.5
        )

        ax.set_title(dataset)
        ax.set_xlabel("Strategy")
        ax.set_ylabel("Count")
        ax.set_ylim(0, max_count + 1)
        ax.tick_params(axis='x', rotation=40)
        ax.get_legend().remove()  

    for j in range(len(datasets), len(axes)):
        axes[j].set_visible(False)

    handles = [plt.Rectangle((0,0),1,1, color=present_colors[label]) for label in present_colors]
    labels = list(present_colors.keys())
    fig.legend(handles, labels, title="Score", loc='upper right', bbox_to_anchor=(1.02, 1), ncol=1)

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    left, center, right = st.columns([1, 8, 1])
    with center:
        st.pyplot(fig)



def plot_confidence(df, score_col="score", confidence_col="confidence",
                    dataset_col="dataset_name", strategy_col="strategy_name"):
    
    st.subheader("Model Confidence Distribution Across Answer Scores")

    if df.empty:
        st.info("No valid data to display.")
        return

    df = df.dropna(subset=[strategy_col, confidence_col])
    datasets = [ds for ds in df[dataset_col].dropna().unique() if not df[df[dataset_col]==ds].empty]
    if not datasets:
        st.info("No datasets with valid data.")
        return

    label_color_map = {
        "Right": "#95cd59",
        "Somewhat right": "#f3be20",
        "Somewhat wrong": "#f38320",
        "Wrong": "#d62728",
        "Unclear": "#7f7f7f",
        "No answer provided": "#69aee3"
    }

    present_labels = df[score_col].dropna().unique().tolist()
    present_colors = {label: label_color_map[label] for label in present_labels if label in label_color_map}

    ncols = 2
    nrows = (len(datasets) + 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*6, nrows*5))
    axes = np.array(axes).flatten() if isinstance(axes, np.ndarray) else np.array([axes])

    for i, dataset in enumerate(datasets):
        df_ds = df[df[dataset_col] == dataset]

        if df_ds.empty or df_ds[strategy_col].dropna().empty or df_ds[confidence_col].dropna().empty:
            axes[i].set_visible(False)
            continue

        ax = axes[i]
        sns.boxplot(
            data=df_ds,
            x=strategy_col,
            y=confidence_col,
            hue=score_col,
            palette=present_colors,
            ax=ax
        )

        ax.set_title(dataset)
        ax.set_xlabel("Strategy")
        ax.set_ylabel("Confidence")
        ax.tick_params(axis='x', rotation=40)
        ax.get_legend().remove() 

    for j in range(len(datasets), len(axes)):
        axes[j].set_visible(False)

    handles = [plt.Rectangle((0,0),1,1, color=present_colors[label]) for label in present_colors]
    labels = list(present_colors.keys())
    fig.legend(handles, labels, title="Score", loc='upper right', bbox_to_anchor=(1.02, 1), ncol=1)

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    left, center, right = st.columns([1, 8, 1])
    with center:
        st.pyplot(fig)


def show_chosen_problem(df, problem_id, dataset_name, strategy_name, strategy_col='strategy_name'):
    if df.empty:
        st.info("No valid data to display.")
        return

    row = df[(df['problem_id'] == problem_id) & (df['dataset_name'] == dataset_name) & (df[strategy_col] == strategy_name)].squeeze()

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
        st.markdown("#### Model Answer")
        st.markdown(
            f"<p style='font-size:18px;'>{_safe_display(row.get('answer'))}</p>",
            unsafe_allow_html=True
        )

        st.markdown("#### Confidence")
        st.markdown(
            f"<p style='font-size:18px;'>{_safe_display(row.get('confidence'))}</p>",
            unsafe_allow_html=True
        )

        st.markdown("#### Key answer")
        st.markdown(
            f"<p style='font-size:18px;'>{_safe_display(row.get('key'))}</p>",
            unsafe_allow_html=True
        )

        st.markdown("#### Score")
        st.markdown(
            f"<p style='font-size:18px;'>{_safe_display(row.get('score'))}</p>",
            unsafe_allow_html=True
        )

        st.markdown("#### Rationale")
        st.markdown(
            f"<p style='font-size:18px;'>{_safe_display(row.get('rationale'))}</p>",
            unsafe_allow_html=True
        )

def display_evaluation_summary(df, dataset_name, strategy_name, strategy_col='strategy_name', is_ensemble=False):
    df = df[(df['dataset_name'] == dataset_name) & (df[strategy_col] == strategy_name)]
    
    model_name = df['model_name'].iloc[0]

    if is_ensemble:
        base_path = os.path.join(
            "results",
            "ensembles",
            df['dataset_name'].iloc[0],
            strategy_name,
            f"ensemble_ver{df['version'].iloc[0]}",
        )
    else:
        base_path = os.path.join(
            "results",
            df['dataset_name'].iloc[0],
            strategy_name,
            shorten_model_name(model_name),
            f"ver{df['version'].iloc[0]}",
        )

    results_metrics_path = os.path.join(base_path, "evaluation_results_metrics.json")
    results_summary_path = os.path.join(base_path, "evaluation_results_summary.json")

    metrics = load_json_safe(results_metrics_path)
    summary = load_json_safe(results_summary_path)

    if metrics is None:
        st.warning(
            f"Metrics not found for strategy '{strategy_name}':\n`{results_metrics_path}`"
        )
        return

    total = metrics.get("total", 0)
    accuracy = metrics.get("accuracy", None)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<h3 style='font-size:24px;;'>Total samples: {_safe_display(total)}</h3>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<h3 style='font-size:24px;;'>Accuracy: {_safe_display(accuracy):.2%}</h3>", unsafe_allow_html=True)

    if not is_ensemble:
        cols = st.columns([3, 1, 2, 2])
        with cols[0]:
            st.markdown("<p style='font-size:18px; font-weight:bold;'>Score</p>", unsafe_allow_html=True)
        with cols[1]:
            st.markdown("<p style='font-size:18px; font-weight:bold;'>Count</p>", unsafe_allow_html=True)
        with cols[2]:
            st.markdown("<p style='font-size:18px; font-weight:bold;'>Average Confidence</p>", unsafe_allow_html=True)
        with cols[3]:
            st.markdown("<p style='font-size:18px; font-weight:bold;'>Median Confidence</p>", unsafe_allow_html=True)

    bin_counts = metrics.get("bin_counts", {})
    avg_conf = metrics.get("avg_confidence")
    med_conf = metrics.get("median_confidence")

    has_confidence = isinstance(avg_conf, dict) and avg_conf

    for label, count in bin_counts.items():
        cols = st.columns([3, 1, 2, 2])

        with cols[0]:
            st.markdown(f"<p style='font-size:18px;'>{_safe_display(label)}</p>", unsafe_allow_html=True)

        with cols[1]:
            st.markdown(f"<p style='font-size:18px;'>{_safe_display(count)}</p>", unsafe_allow_html=True)

        with cols[2]:
            if has_confidence:
                val = round(avg_conf.get(label), 2)
                st.markdown(
                    f"<p style='font-size:18px;'>{_safe_display(val)}</p>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown("<p style='font-size:18px;'>–</p>", unsafe_allow_html=True)

        with cols[3]:
            if has_confidence:
                val = round(med_conf.get(label), 2)
                st.markdown(
                    f"<p style='font-size:18px;'>{_safe_display(val)}</p>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown("<p style='font-size:18px;'>–</p>", unsafe_allow_html=True)


    if summary:
        st.markdown("<p style='font-size:18px; font-weight:bold;'>Data completeness</p>", unsafe_allow_html=True)

        cols = st.columns([3, 2, 2, 2])
        with cols[0]:
            st.markdown("<p style='font-size:14px; color:gray;'></p>", unsafe_allow_html=True)
        with cols[1]:
            st.markdown("<p style='font-size:14px; color:gray;'>Number of samples present</p>", unsafe_allow_html=True)
        with cols[2]:
            st.markdown("<p style='font-size:14px; color:gray;'>Fraction of present samples</p>", unsafe_allow_html=True)
        with cols[3]:
            st.markdown("<p style='font-size:14px; color:gray;'>Number of missing samples</p>", unsafe_allow_html=True)

        for section_key, section_label in [
            ("answers_completeness", "Answers"),
            ("key_completeness", "Answer key"),
        ]:
            section = summary.get(section_key)
            if not section:
                continue

            expected = section.get("expected_num_samples", 0)
            missing = section.get("num_missing_problem_ids", 0)
            available = expected - missing
            coverage = (available / expected) if expected else 0.0

            cols = st.columns([3, 2, 2, 2])
            with cols[0]:
                st.markdown(f"<p style='font-size:18px; font-weight:bold;'>{_safe_display(section_label)}</p>", unsafe_allow_html=True)
            with cols[1]:
                st.markdown(f"<p style='font-size:18px;'>{_safe_display(available)} / {_safe_display(expected)}</p>", unsafe_allow_html=True)
            with cols[2]:
                st.markdown(f"<p style='font-size:18px;'>{_safe_display(f'{coverage:.1%}')}</p>", unsafe_allow_html=True)
            with cols[3]:
                st.markdown(f"<p style='font-size:18px;'>{_safe_display(f'{missing} missing')}</p>", unsafe_allow_html=True)

def load_json_safe(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Failed to load JSON: {e}")
        return None

def show_single_model_config(
    model_name,
    dataset_name,
    strategy_name,
    version
):
    shortened_model_name = shorten_model_name(model_name)
    st.subheader("Single Model Configuration")

    tech_model_config_path = os.path.join(
        "src", 
        "technical", 
        "configs", 
        "models_config.json"
    )

    metadata_path = os.path.join(
        "results",
        dataset_name,
        strategy_name,
        shortened_model_name,
        f"ver{version}",
        "metadata.json",
    )

    metadata = load_json_safe(metadata_path)
    if metadata is None:
        st.warning(f"Metadata not found:\n`{metadata_path}`")
        return

    tech_config = load_json_safe(tech_model_config_path)
    if tech_config is None:
        st.warning(f"Model config not found:\n`{tech_model_config_path}`")
        return
    model_entry = tech_config.get(model_name, {})
    param_sets = model_entry.get("param_sets", {})

    temperature = None
    model_param_set = metadata.get("param_set_number")
    if model_param_set and str(model_param_set) in param_sets:
        temperature = param_sets[str(model_param_set)].get("temperature")

    st.markdown(f"<p style='font-size:18px;'><b>Model name:</b> {metadata.get('model')}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:18px;'><b>Temperature:</b> {temperature}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:18px;'><b>Dataset:</b> {metadata.get('dataset')}</p>", unsafe_allow_html=True)
    category = metadata.get("config", {}).get("category")
    st.markdown(f"<p style='font-size:18px;'><b>Dataset Category:</b> {category}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:18px;'><b>Strategy:</b> {metadata.get('strategy')}</p>", unsafe_allow_html=True)


    st.markdown("### Prompts")

    st.text_area(
        "Problem Description Prompt",
        value=metadata.get("problem_description_prompt", ""),
        height=150,
    )

    st.text_area(
        "Question Prompt",
        value=metadata.get("question_prompt", ""),
        height=180,
    )

    if metadata.get("sample_answer_prompt"):
        st.text_area(
            "Sample Answer Prompt",
            value=metadata.get("sample_answer_prompt", ""),
            height=100,
        )

    
    with st.expander("Show full experiment metadata"):
        st.json(metadata)

    with st.expander("Show full technical model config"):
        if model_entry:
            st.json(model_entry)
        else:
            st.info("No technical config found for this model.")


def show_ensemble_config(dataset_name, type_name, ensemble_version):
    st.subheader("Ensemble Configuration")

    metadata_path = os.path.join(
        "results",
        "ensembles",
        dataset_name,
        type_name,
        f"ensemble_ver{ensemble_version}",
        "ensemble_config.json",
    )

    metadata = load_json_safe(metadata_path)
    if metadata is None:
        st.warning(f"Metadata not found:\n`{metadata_path}`")
        return
    
    tech_model_config_path = os.path.join(
        "src", 
        "technical", 
        "configs", 
        "models_config.json"
    )
    tech_config = load_json_safe(tech_model_config_path)
    if tech_config is None:
        st.warning(f"Model config not found:\n`{tech_model_config_path}`")
        return

    st.markdown(f"<p style='font-size:18px;'><b>Ensemble Model:</b> {metadata.get('ensemble_model')}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:18px;'><b>Main Prompt:</b> {metadata.get('main_prompt')}</p>", unsafe_allow_html=True)

    member_keys = sorted([k for k in metadata.keys() if k.startswith("member_")])
    for member_key in member_keys:
        member = metadata[member_key]
        st.markdown(f"#### {member_key.capitalize().replace('_', ' ')}")
        model_name = member.get("model")
        st.write(f"**Model Name:** {model_name}")
        model_entry = tech_config.get(model_name, {})
        st.write(f"**Technical Config Entry:** {model_entry}")
        param_sets = model_entry.get("param_sets", {})
        st.write(f"**Parameter Sets:** {param_sets}")
        temperature = None
        model_param_set = member.get("param_set_number")
        
        if model_param_set and str(model_param_set) in param_sets:
            temperature = param_sets[str(model_param_set)].get("temperature")
        st.markdown(f"<p style='font-size:18px;'><b>Model:</b> {model_name}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:18px;'><b>Temperature:</b> {temperature}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:18px;'><b>Dataset:</b> {member.get('dataset')}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:18px;'><b>Strategy:</b> {member.get('strategy')}</p>", unsafe_allow_html=True)
        category = member.get("config", {}).get("category")
        st.markdown(f"<p style='font-size:18px;'><b>Dataset Category:</b> {category}</p>", unsafe_allow_html=True)

        with st.expander(f"Show full config for {member_key.capitalize().replace('_', ' ')}"):
            st.json(member)

    st.markdown(f"#### Full Ensemble metadata")
    with st.expander("Show full ensemble metadata"):
        st.json(metadata)


def show_problem_strategy_table(
    df,
    dataset_name,
    outcome_col = "score",
    problem_col = "problem_id",
    strategy_col = "strategy_name"
):

    st.subheader(f"Problem × Strategy Outcome Overview")

    df = df[df['dataset_name'] == dataset_name].copy()
    if df.empty:
        st.info("No data for selected dataset.")
        return

    label_color_map = {
        "Right": "#95cd59",
        "Somewhat right": "#f3be20",
        "Somewhat wrong": "#f38320",
        "Wrong": "#d62728",
        "Unclear": "#7f7f7f",
        "No answer provided": "#69aee3"
    }

    df[outcome_col] = df[outcome_col].fillna("No answer provided")

    pivot = df.pivot(
        index=problem_col,
        columns=strategy_col,
        values=outcome_col
    ).sort_index()

    pivot = pivot.fillna("")

    def color_cells(val):
        if val in label_color_map:
            return f"background-color: {label_color_map[val]}; color: black"
        else:
            return ""

    styled = pivot.style.applymap(color_cells)

    st.dataframe(
        styled,
        height=450,
        use_container_width=True
    )
