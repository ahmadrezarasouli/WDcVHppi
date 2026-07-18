import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def calculate_metrics(y_true, y_pred, y_prob):

    return {

        "Accuracy": accuracy_score(y_true, y_pred),

        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "Recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "F1-score": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "ROC-AUC": roc_auc_score(
            y_true,
            y_prob,
        ),

        "PR-AUC": average_precision_score(
            y_true,
            y_prob,
        ),
    }


def metrics_dataframe(metrics):

    return pd.DataFrame([metrics])


def get_confusion_matrix(
    y_true,
    y_pred,
):

    return confusion_matrix(
        y_true,
        y_pred,
    )


def get_classification_report(
    y_true,
    y_pred,
):

    return classification_report(
        y_true,
        y_pred,
        zero_division=0,
    )
