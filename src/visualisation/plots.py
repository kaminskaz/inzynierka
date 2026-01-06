import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.visualisation.data_handling import get_present_colors

def create_countplot(df, score_col="score", dataset_col="dataset_name", strategy_col="strategy_name"):
    datasets = df[dataset_col].dropna().unique()
    if len(datasets) == 0:
        return None

    colors = get_present_colors(df, score_col)
    ncols = 2
    nrows = (len(datasets) + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*4))
    axes = np.array(axes).flatten() if isinstance(axes, np.ndarray) else np.array([axes])

    max_count = df.groupby([dataset_col, strategy_col, score_col]).size().max()

    for i, dataset in enumerate(datasets):
        df_ds = df[df[dataset_col] == dataset]
        if df_ds.empty or df_ds[strategy_col].dropna().empty:
            axes[i].set_visible(False)
            continue
        sns.countplot(
            data=df_ds,
            x=strategy_col,
            hue=score_col,
            palette=colors,
            ax=axes[i],
            order=sorted(df_ds[strategy_col].unique()),
            width=0.5
        )
        axes[i].set_title(dataset)
        axes[i].set_xlabel("Strategy")
        axes[i].set_ylabel("Count")
        axes[i].set_ylim(0, max_count + 1)
        axes[i].tick_params(axis='x', rotation=40)
        axes[i].get_legend().remove()

    for j in range(len(datasets), len(axes)):
        axes[j].set_visible(False)

    handles = [plt.Rectangle((0,0),1,1,color=colors[label]) for label in colors]
    labels = list(colors.keys())
    fig.legend(handles, labels, title="Score", loc='upper right', bbox_to_anchor=(1.02,1))
    plt.tight_layout(rect=[0,0,0.8,1])
    return fig

def create_confidence_boxplot(df, score_col="score", confidence_col="confidence",
                              dataset_col="dataset_name", strategy_col="strategy_name"):
    datasets = df[dataset_col].dropna().unique()
    if len(datasets) == 0:
        return None
    if df[confidence_col].dropna().empty:
        return None
    colors = get_present_colors(df, score_col)
    ncols = 2
    nrows = (len(datasets) + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*4))
    axes = np.array(axes).flatten() if isinstance(axes, np.ndarray) else np.array([axes])

    min_confidence = df[confidence_col].min() if not df[confidence_col].empty else 0.0
    max_confidence = df[confidence_col].max() if not df[confidence_col].empty else 1.0

    for i, dataset in enumerate(datasets):
        df_ds = df[df[dataset_col] == dataset]
        if df_ds.empty or df_ds[strategy_col].dropna().empty or df_ds[confidence_col].dropna().empty:
            axes[i].set_visible(False)
            continue
        sns.boxplot(
            data=df_ds,
            x=strategy_col,
            y=confidence_col,
            hue=score_col,
            palette=colors,
            ax=axes[i]
        )
        axes[i].set_title(dataset)
        axes[i].set_xlabel("Strategy")
        axes[i].set_ylim(min_confidence - 0.05, max_confidence + 0.05)
        axes[i].set_ylabel("Confidence")
        axes[i].tick_params(axis='x', rotation=40)
        axes[i].get_legend().remove()

    for j in range(len(datasets), len(axes)):
        axes[j].set_visible(False)

    handles = [plt.Rectangle((0,0),1,1,color=colors[label]) for label in colors]
    labels = list(colors.keys())
    fig.legend(handles, labels, title="Score", loc='upper right', bbox_to_anchor=(1.02,1))
    plt.tight_layout(rect=[0,0,0.8,1])
    return fig
