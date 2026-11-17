"""
visualizations.py - Functions that build Matplotlib Figures.

Figure objects (not pyplot) are used so they can be embedded in Tkinter.
"""

import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from sklearn.tree import plot_tree

MALIGNANT_COLOR = "#c0392b"
BENIGN_COLOR = "#27ae60"


def class_distribution_figure(df, class_names):
    """Bar chart of benign vs malignant sample counts."""
    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)
    counts = df["target"].value_counts().sort_index()      # index 0, 1
    bars = ax.bar(class_names, counts.values,
                  color=[MALIGNANT_COLOR, BENIGN_COLOR])
    for bar, c in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, c + 4, str(c),
                ha="center", fontweight="bold")
    ax.set_title("Class Distribution")
    ax.set_ylabel("Number of samples")
    ax.set_ylim(0, counts.max() * 1.15)
    fig.tight_layout()
    return fig


def confusion_matrix_figure(cm, class_names):
    """Seaborn heatmap of the confusion matrix."""
    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 16})
    ax.set_title("Confusion Matrix (Test Set)")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    fig.tight_layout()
    return fig


def decision_tree_figure(clf, feature_names, class_names, size=(16, 8)):
    """The trained CART tree. Each node shows its Gini value."""
    fig = Figure(figsize=size, dpi=100)
    FigureCanvasAgg(fig)   # plot_tree needs a canvas that can render text
    ax = fig.add_subplot(111)
    plot_tree(clf, feature_names=feature_names, class_names=class_names,
              filled=True, rounded=True, impurity=True, fontsize=7, ax=ax)
    ax.set_title("CART Decision Tree (Gini criterion)")
    fig.tight_layout()
    return fig
