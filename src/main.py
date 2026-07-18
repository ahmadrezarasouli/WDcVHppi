from data.load_data import load_interactions
from preprocessing.clean_data import clean_interactions
from embedding.sequence_embedding import generate_sequence_embeddings
from embedding.graph_embedding import generate_graph_embeddings
from preprocessing.negative_sampling import generate_negative_samples
from preprocessing.prepare_dataset import prepare_dataset

from training.preprocessing import prepare_train_test_data

from training.wide_and_deep import (
    build_model as build_wd,
    train_model as train_wd,
    predict as predict_wd,
)

from evaluation.metrics import (
    calculate_metrics,
    metrics_dataframe,
)

from evaluation.plots import (
    plot_roc_curve,
    plot_pr_curve,
    plot_confusion_matrix,
)


def main():

    interactions = load_interactions()

    interactions = clean_interactions(interactions)

    sequence_features = generate_sequence_embeddings()

    graph_features = generate_graph_embeddings()

    negative_samples = generate_negative_samples(interactions)

    dataset = prepare_dataset(
        interactions,
        negative_samples,
        sequence_features,
        graph_features,
    )

    X_train, X_test, y_train, y_test = prepare_train_test_data(dataset)

    model = build_wd(X_train.shape[1])

    history = train_wd(
        model,
        X_train,
        y_train,
    )

    y_pred, y_prob = predict_wd(
        model,
        X_test,
    )

    metrics = calculate_metrics(
        y_test,
        y_pred,
        y_prob,
    )

    print(metrics_dataframe(metrics))

    plot_roc_curve(
        y_test,
        y_prob,
    )

    plot_pr_curve(
        y_test,
        y_prob,
    )

    plot_confusion_matrix(
        y_test,
        y_pred,
    )


if __name__ == "__main__":
    main()
