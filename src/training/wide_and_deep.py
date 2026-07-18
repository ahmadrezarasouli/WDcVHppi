import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping


def build_model(input_dim):

    inputs = Input(shape=(input_dim,))

    deep = Dense(256, activation="relu")(inputs)
    deep = Dropout(0.2)(deep)

    deep = Dense(128, activation="relu")(deep)
    deep = Dropout(0.2)(deep)

    deep = Dense(128, activation="relu")(deep)
    deep = Dropout(0.2)(deep)

    deep = Dense(64, activation="relu")(deep)
    deep = Dropout(0.2)(deep)

    deep = Dense(32, activation="relu")(deep)
    deep = Dropout(0.2)(deep)

    wide_deep = Concatenate()([inputs, deep])

    outputs = Dense(1, activation="sigmoid")(wide_deep)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_model(
    model,
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
):

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1,
    )

    return history


def predict(model, X):

    probabilities = model.predict(X).flatten()

    predictions = (probabilities >= 0.5).astype(int)

    return predictions, probabilities
