import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


def plot_roc_curve(
    y_true,
    y_prob,
    save_path=None,
):

    fpr, tpr, _ = roc_curve(
        y_true,
        y_prob,
    )

    roc_auc = auc(
        fpr,
        tpr,
    )

    plt.figure(figsize=(7, 6))

    plt.plot(
        fpr,
        tpr,
        lw=2,
        label=f"AUC = {roc_auc:.4f}",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        "--",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()


def plot_pr_curve(
    y_true,
    y_prob,
    save_path=None,
):

    precision, recall, _ = precision_recall_curve(
        y_true,
        y_prob,
    )

    pr_auc = auc(
        recall,
        precision,
    )

    plt.figure(figsize=(7, 6))

    plt.plot(
        recall,
        precision,
        lw=2,
        label=f"AUC = {pr_auc:.4f}",
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()

    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()


def plot_confusion_matrix(
    y_true,
    y_pred,
    save_path=None,
):

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
    )

    disp.plot()

    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()


def plot_training_history(
    history,
    save_path=None,
):

    plt.figure(figsize=(8, 6))

    plt.plot(
        history.history["loss"],
        label="Train Loss",
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.legend()

    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()
