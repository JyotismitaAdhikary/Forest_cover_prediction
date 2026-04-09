import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def forest_cover_trend_fig(cover_df):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        cover_df["year"], cover_df["forest_cover_pct"],
        marker="o", linewidth=2.5, color="#2d6a4f", markersize=8,
    )
    ax.fill_between(
        cover_df["year"], cover_df["forest_cover_pct"],
        alpha=0.15, color="#2d6a4f",
    )
    ax.set_title("Assam Forest Cover Trend", fontsize=14, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Forest Cover (%)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def feature_importance_fig(importance_df, model_name):
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#2d6a4f" if i < 3 else "#95d5b2" for i in range(len(importance_df))]
    ax.barh(
        importance_df["feature"][::-1],
        importance_df["importance"][::-1],
        color=colors[::-1],
    )
    ax.set_title(f"Feature Importance — {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    return fig


def ndvi_distribution_fig(df):
    fig, ax = plt.subplots(figsize=(9, 4))
    for label, color, name in [(1, "#2d6a4f", "Forest"), (0, "#d4a373", "Non-Forest")]:
        subset = df[df["label"] == label]["NDVI"].dropna()
        ax.hist(subset, bins=60, alpha=0.6, color=color, label=name, density=True)
    ax.set_title("NDVI Distribution: Forest vs Non-Forest", fontsize=13, fontweight="bold")
    ax.set_xlabel("NDVI")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    return fig


def confusion_matrix_fig(cm, model_name):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-Forest", "Forest"])
    ax.set_yticklabels(["Non-Forest", "Forest"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}", fontweight="bold")
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=13,
            )
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig


def roc_curve_fig(y_test, results_dict):
    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = {"Random Forest": "#2d6a4f", "LightGBM": "#52b788"}
    for name, res in results_dict.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']})", color=colors[name], lw=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    return fig

