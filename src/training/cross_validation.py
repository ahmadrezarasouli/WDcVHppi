import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


def cross_validate(
    build_model,
    train_model,
    predict,
    X,
    y,
    n_splits=5,
):

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    results = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        model = build_model(X_train.shape[1])

        model = train_model(
            model,
            X_train,
            y_train,
        )

        y_pred, y_prob = predict(
            model,
            X_test,
        )

        results.append({

            "Fold": fold,

            "Accuracy":
            accuracy_score(y_test, y_pred),

            "Precision":
            precision_score(
                y_test,
                y_pred,
                zero_division=0,
            ),

            "Recall":
            recall_score(
                y_test,
                y_pred,
                zero_division=0,
            ),

            "F1":
            f1_score(
                y_test,
                y_pred,
                zero_division=0,
            ),

            "ROC_AUC":
            roc_auc_score(
                y_test,
                y_prob,
            ),

            "PR_AUC":
            average_precision_score(
                y_test,
                y_prob,
            ),

        })

    return results
