from xgboost import XGBClassifier


def build_model(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
):

    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        eval_metric="logloss",
        random_state=random_state,
    )

    return model


def train_model(
    model,
    X_train,
    y_train,
):

    model.fit(
        X_train,
        y_train,
    )

    return model


def predict(
    model,
    X_test,
):

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = model.predict(X_test)

    return predictions, probabilities
