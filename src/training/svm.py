from sklearn.svm import SVC


def build_model(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True,
    random_state=42,
):

    model = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        probability=probability,
        random_state=random_state,
    )

    return model


def train_model(
    model,
    X_train,
    y_train,
):

    model.fit(X_train, y_train)

    return model


def predict(
    model,
    X_test,
):

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = model.predict(X_test)

    return predictions, probabilities
