from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def create_data_visualizations(cleaned_df, plots_dir_path):
    plots_dir = Path(plots_dir_path)
    plots_dir.mkdir(parents=True, exist_ok=True)

    target_column = "outcome" if "outcome" in cleaned_df.columns else "class"

    plt.figure(figsize=(7, 5))
    sns.countplot(x=target_column, data=cleaned_df, palette="Set2")
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(plots_dir / "class_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 8))
    correlation_matrix = cleaned_df.corr(numeric_only=True)
    sns.heatmap(correlation_matrix, cmap="coolwarm", center=0, annot=False)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(plots_dir / "correlation_heatmap.png", dpi=150)
    plt.close()


def plot_model_metrics(metrics, plots_dir_path):
    plots_dir = Path(plots_dir_path)
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame.from_dict(metrics, orient="index")
    metrics_df.index.name = "model"
    metrics_df = metrics_df.reset_index()

    plot_data = metrics_df.melt(id_vars="model", var_name="metric", value_name="score")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_data, x="model", y="score", hue="metric")
    plt.title("Model Performance Comparison")
    plt.ylim(0, 1)
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(plots_dir / "model_performance.png", dpi=150)
    plt.close()