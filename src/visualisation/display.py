import os
import json
import streamlit as st

from src.visualisation.plots import (
    create_countplot,
    create_confidence_boxplot
)
from src.visualisation.data_handling import get_present_colors
from src.visualisation.ui_helpers import shorten_model_name, safe_display


# ======================================================
# Helpers
# ======================================================

def load_json_safe(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Failed to load JSON: {e}")
        return None


def render_markdown(label: str, value, font_size=18):
    """Label–value renderer for config-style sections."""
    st.markdown(
        f"<p style='font-size:{font_size}px;'>"
        f"<b>{safe_display(label)}</b>: {safe_display(value)}"
        f"</p>",
        unsafe_allow_html=True
    )


def render_cell(value, font_size=18, bold=False):
    """HTML-safe table/grid cell renderer."""
    weight = "font-weight:bold;" if bold else ""
    st.markdown(
        f"<p style='{weight} font-size:{font_size}px;'>"
        f"{safe_display(value)}</p>",
        unsafe_allow_html=True
    )


def centered_st(func):
    def wrapper(*args, **kwargs):
        _, center, _ = st.columns([1, 8, 1])
        with center:
            return func(*args, **kwargs)
    return wrapper


# ======================================================
# Plotting
# ======================================================

@centered_st
def plot_judged_answers(df, **kwargs):
    st.subheader("Scored Answers Summary")
    fig = create_countplot(df, **kwargs)
    st.pyplot(fig) if fig else st.info("No valid data to display.")


@centered_st
def plot_confidence(df, **kwargs):
    st.subheader("Model Confidence Distribution Across Answer Scores")
    fig = create_confidence_boxplot(df, **kwargs)
    st.pyplot(fig) if fig else st.info("No valid data to display.")


# ======================================================
# Problem Display
# ======================================================

def show_chosen_problem(df, problem_id, dataset_name, strategy_name, strategy_col="strategy_name"):
    if df.empty:
        st.info("No valid data to display.")
        return

    row = df[
        (df["problem_id"] == problem_id)
        & (df["dataset_name"] == dataset_name)
        & (df[strategy_col] == strategy_name)
    ].squeeze()

    col1, col2 = st.columns(2)

    with col1:
        img_path = os.path.join(
            "data", dataset_name, "problems", str(problem_id), "question_panel.png"
        ) if dataset_name else None

        if img_path and os.path.exists(img_path):
            st.image(img_path, caption=f"{dataset_name} – Problem {problem_id}")
        else:
            st.warning("Problem image not available.")

    with col2:
        for k in ["answer", "confidence", "key", "score", "rationale"]:
            render_markdown(k.replace("_", " ").title(), row.get(k))


# ======================================================
# Evaluation Summary
# ======================================================

def display_evaluation_summary(df, dataset_name, strategy_name,
                               strategy_col="strategy_name", is_ensemble=False):

    df = df[
        (df["dataset_name"] == dataset_name)
        & (df[strategy_col] == strategy_name)
    ]

    if df.empty:
        st.info("No data for this dataset/strategy.")
        return

    model_name = df["model_name"].iloc[0]
    version = df["version"].iloc[0]

    base_path = (
        os.path.join("results", "ensembles", dataset_name, strategy_name, f"ensemble_ver{version}")
        if is_ensemble else
        os.path.join("results", dataset_name, strategy_name,
                     shorten_model_name(model_name), f"ver{version}")
    )

    metrics = load_json_safe(os.path.join(base_path, "evaluation_results_metrics.json"))
    summary = load_json_safe(os.path.join(base_path, "evaluation_results_summary.json"))

    if not metrics:
        st.warning(f"Metrics not found at {base_path}")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<h3>Total samples: {safe_display(metrics.get('total', 0))}</h3>",
                    unsafe_allow_html=True)
    
    with col2:
        if dataset_name.lower() == "bp" and strategy_name.lower() != "classification":
            points_map = {
                "right": 1.0,
                "somewhat right": 0.5
            }

            total_points = 0
            total_count = 0
            for label, count in metrics.get("bin_counts", {}).items():
                pts_per_item = points_map.get(label.lower(), 0)
                total_points += pts_per_item * count
                total_count += count

            st.markdown(f"<h3>Points: {safe_display(total_points):.2f}</h3>", unsafe_allow_html=True)

        else:
            st.markdown(f"<h3>Accuracy: {safe_display(metrics.get('accuracy')):.2%}</h3>",
                        unsafe_allow_html=True)

    if not is_ensemble:
        headers = ["Score", "Count", "Average Confidence", "Median Confidence"]
        for c, h in zip(st.columns([3, 1, 2, 2]), headers):
            with c:
                render_cell(h, bold=True)

    for label, count in metrics.get("bin_counts", {}).items():
        avg = metrics.get("avg_confidence", {}).get(label, 0)
        med = metrics.get("median_confidence", {}).get(label, 0)

        cols = st.columns([3, 2, 2, 2])
        with cols[0]: render_cell(label)
        with cols[1]: render_cell(count)
        with cols[2]: render_cell(round(avg, 2))
        with cols[3]: render_cell(round(med, 2))

    if summary:
        render_cell("Data completeness", bold=True) 

        for key, label in [("answers_completeness", "Answers"), ("key_completeness", "Answer key")]:
            section = summary.get(key)
            if not section:
                continue

            expected = section.get("expected_num_samples", 0)
            missing = section.get("num_missing_problem_ids", 0)
            available = expected - missing
            coverage = available / expected if expected else 0

            cols = st.columns([3, 2, 2, 2])
            with cols[0]: render_cell(label)
            with cols[1]: render_cell(f"{available}/{expected}")
            with cols[2]: render_cell(f"{coverage:.1%}")
            with cols[3]: render_cell(f"{missing} missing")

            missing_ids = section.get("missing_problem_ids", [])
            if missing_ids:
                st.markdown(
                    f"<p style='font-size:18px; color:#d62728;'>"
                    f"<i>Missing Problem IDs: {', '.join(missing_ids)}</i></p>",
                    unsafe_allow_html=True
                )
    

# ======================================================
# Single Model Configuration
# ======================================================

def show_single_model_config(model_name, dataset_name, strategy_name, version):
    st.subheader("Configuration")

    base = os.path.join("results", dataset_name, strategy_name,
                        shorten_model_name(model_name), f"ver{version}")
    metadata = load_json_safe(os.path.join(base, "metadata.json"))
    if not metadata:
        st.warning("Metadata not found.")
        return

    tech_cfg = load_json_safe(os.path.join("src", "technical", "configs", "models_config.json")) or {}
    param_sets = tech_cfg.get(model_name, {}).get("param_sets", {})

    ps = metadata.get("param_set_number")
    temperature = param_sets.get(str(ps), {}).get("temperature") if ps else None

    render_markdown("Model name", metadata.get("model"))
    render_markdown("Temperature", temperature)
    render_markdown("Dataset", metadata.get("dataset"))
    render_markdown("Dataset Category", metadata.get("config", {}).get("category"))
    render_markdown("Task Type", metadata.get("config", {}).get("task_type"))
    render_markdown("Strategy", metadata.get("strategy"))

    st.markdown("### Prompts")
    prompts = [
        ("problem_description_prompt", 150),
        ("question_prompt", 180),
        ("sample_answer_prompt", 100),
        ("example_prompt", 150),
        ("describe_prompt", 150),
        ("describe_example_prompt", 150),
        ("contrast_example_prompt", 150)
    ]

    for key, height in prompts:
        value = metadata.get(key)
        if value:
            st.text_area(key.replace("_", " ").title(), value, height=height)


    with st.expander("Full experiment metadata"):
        st.json(metadata)

    with st.expander("Full technical model config"):
        st.json(tech_cfg.get(model_name, {}))


# ======================================================
# Ensemble Configuration
# ======================================================

def show_ensemble_config(dataset_name, type_name, ensemble_version):
    st.subheader("Ensemble Configuration")

    base = os.path.join("results", "ensembles", dataset_name, type_name,
                        f"ensemble_ver{ensemble_version}")
    metadata = load_json_safe(os.path.join(base, "ensemble_config.json"))
    if not metadata:
        st.warning("Metadata not found.")
        return

    tech_cfg = load_json_safe(os.path.join("src", "technical", "configs", "models_config.json")) or {}

    for k in ["ensemble_model", "dataset_name", "dataset_category", "task_type"]:
        render_markdown(k.replace("_"," ").title(), metadata.get(k))

    st.markdown("### Main Prompt")
    st.text_area("Main Prompt", safe_display(metadata.get("main_prompt", "")), height=300)

    for mk in sorted(k for k in metadata if k.startswith("member_")):
        member = metadata[mk]
        st.markdown(f"#### {mk.replace('_',' ').title()}")

        model = member.get("model")
        param_sets = tech_cfg.get(model, {}).get("param_sets", {})
        ps = member.get("param_set_number")
        temperature = param_sets.get(str(ps), {}).get("temperature") if ps else None

        render_markdown("Model Name", model)
        render_markdown("Temperature", temperature)
        render_markdown("Strategy", member.get("strategy"))

        with st.expander(f"Full config for {mk.replace('_',' ').title()}"):
            st.json(member)

    with st.expander("Full Ensemble metadata"):
        st.json(metadata)


# ======================================================
# Problem × Strategy Table
# ======================================================

def show_problem_strategy_table(df, dataset_name,
                                outcome_col="score",
                                problem_col="problem_id",
                                strategy_col="strategy_name"):

    st.subheader("Problem × Strategy Outcome Overview")

    df = df[df["dataset_name"] == dataset_name].copy()
    if df.empty:
        st.info("No data for selected dataset.")
        return

    df[outcome_col] = df[outcome_col].fillna("No answer provided")
    color_map = get_present_colors(df, score_col=outcome_col)

    pivot = (
        df.pivot(index=problem_col, columns=strategy_col, values=outcome_col)
        .sort_index()
        .fillna("")
    )

    styled = pivot.style.applymap(
        lambda v: f"background-color:{color_map[v]}; color:black"
        if v in color_map else ""
    )

    st.dataframe(styled, height=450, use_container_width=True)
