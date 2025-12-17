"""
Neural network Project 1
===================================

Authors:
- Wiktor Rapacz
- Hanna Paczoska

Problem statement:
Train models to predict the cat's gender (Male/Female) using tabular features from a CSV file.
Compare classic machine-learning approaches with a neural network.

Solution overview:
1) Classic ML baseline models:
   - Decision Tree (sklearn)
   - SVM with RBF kernel (sklearn)
2) Neural network model:
   - MLP (Multi-Layer Perceptron) implemented with TensorFlow/Keras (tf.keras)
3) Evaluation:
   - accuracy and classification report for each model
   - confusion matrix for the neural network model

Data source:
- Local CSV file: cats_dataset.csv

How it works (high-level):
- Read CSV into a pandas DataFrame.
- Split into train/test using stratification to keep class proportions.
- Preprocess features:
  - categorical columns: one-hot encoding
  - numeric columns: standard scaling
- Train and evaluate DT, SVM, and MLP.
- Save:
  - log file: outputs/logs/p1_log.txt
  - confusion matrix plot: outputs/plots/p1_confusion_matrix.png

Environment setup (Windows / PyCharm example):
1) Use Python 3.10.x (recommended for TensorFlow 2.10.*).
2) Create and activate virtual environment (example):
   - python -m venv .venv
   - .venv\\Scripts\\activate
3) Install dependencies:
   - pip install -r requirements.txt
4) Run:
   - python p1.py

Notes:
- This script compares classic ML vs a neural network on the same dataset.
- Neural networks are implemented using TensorFlow/Keras (tf.keras).
"""

import os
import sys
import argparse
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models


@dataclass
class TeeLogger:
    """
    A tiny "tee" logger that writes messages both to console and to a text file.

    This is useful for course assignments where the instructor expects logs
    to be saved in the repository (e.g., outputs/logs/*.txt).

    Attributes:
        path: Output path of the log file.
    """

    path: str

    def __post_init__(self) -> None:
        """
        Create the parent directory (if needed) and open the log file.
        """
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._f = open(self.path, "w", encoding="utf-8")

    def print(self, *args) -> None:
        """
        Print text to console and write the same text to the log file.

        Args:
            *args: Values to print (converted to strings and joined by spaces).
        """
        text = " ".join(str(a) for a in args)
        sys.stdout.write(text + "\n")
        self._f.write(text + "\n")
        self._f.flush()

    def close(self) -> None:
        """
        Close the underlying log file safely.
        """
        try:
            self._f.close()
        except (OSError, ValueError):
            # ValueError can happen e.g. if file is already closed
            pass


def save_confusion_matrix_png(cm: np.ndarray, labels: list[str], out_path: str, title: str) -> None:
    """
    Save a confusion matrix plot as a PNG file.

    Args:
        cm: Confusion matrix array, shape (n_classes, n_classes).
        labels: Class labels (displayed on axes).
        out_path: Output PNG path.
        title: Plot title.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def set_seeds(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Integer seed used for Python's random, NumPy, and TensorFlow.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_mlp(input_dim: int, hidden_units: int, hidden_layers: int, dropout: float) -> tf.keras.Model:
    """
    Build and compile a simple MLP model for binary classification.

    Args:
        input_dim: Number of input features after preprocessing (one-hot + scaling).
        hidden_units: Number of neurons in each hidden Dense layer.
        hidden_layers: How many hidden Dense layers to create.
        dropout: Dropout rate (0 disables dropout).

    Returns:
        A compiled tf.keras.Model ready for training.
    """
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    for _ in range(hidden_layers):
        model.add(layers.Dense(hidden_units, activation="relu"))
        if dropout > 0:
            model.add(layers.Dropout(dropout))

    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    """
    Entry point of the script.

    Reads the cats dataset, trains three models (Decision Tree, SVM, MLP),
    compares their performance, and saves logs + confusion matrix plot.

    Outputs:
        - outputs/logs/p1_log.txt
        - outputs/plots/p1_confusion_matrix.png
    """
    parser = argparse.ArgumentParser(description="NAI - Task 5 - Step 1 (cats_dataset.csv)")
    parser.add_argument("--data", default="data/cats_dataset.csv", help="Path to cats_dataset.csv")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set size (default 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--hidden_units", type=int, default=64)
    parser.add_argument("--hidden_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    args = parser.parse_args()

    log_path = "outputs/logs/p1_log.txt"
    cm_path = "outputs/plots/p1_confusion_matrix.png"
    os.makedirs("outputs/logs", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)

    logger = TeeLogger(log_path)
    set_seeds(args.seed)

    logger.print("=== Step 1: cats_dataset.csv | Compare: DT vs SVM vs MLP (Keras) ===")
    logger.print(f"Data: {args.data}")
    logger.print(f"Seed: {args.seed}")
    logger.print(f"Test size: {args.test_size}")
    logger.print(
        "MLP params:",
        f"epochs={args.epochs}, batch={args.batch_size}, hidden_units={args.hidden_units}, "
        f"hidden_layers={args.hidden_layers}, dropout={args.dropout}",
    )
    logger.print("")

    df = pd.read_csv(args.data)

    required_cols = ["Breed", "Age (Years)", "Weight (kg)", "Color", "Gender"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.print("ERROR: Missing columns in CSV:", missing)
        logger.print("Available columns:", list(df.columns))
        logger.close()
        sys.exit(1)

    x = df[["Breed", "Age (Years)", "Weight (kg)", "Color"]].copy()
    y_raw = df["Gender"].astype(str).str.strip()

    y = y_raw.map({"Female": 0, "Male": 1})
    if y.isna().any():
        bad_vals = sorted(y_raw[y.isna()].unique().tolist())
        logger.print("ERROR: Unknown values in 'Gender':", bad_vals)
        logger.close()
        sys.exit(1)

    logger.print(f"Records: {len(df)}")
    logger.print("Class distribution (Gender):")
    logger.print(y_raw.value_counts().to_string())
    logger.print("")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y.values,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y.values,
    )

    cat_cols = ["Breed", "Color"]
    num_cols = ["Age (Years)", "Weight (kg)"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop",
    )

    dt_pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", DecisionTreeClassifier(random_state=args.seed)),
        ]
    )

    svm_pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=args.seed)),
        ]
    )

    logger.print("=== Classic ML: Decision Tree ===")
    dt_pipeline.fit(x_train, y_train)
    dt_pred = dt_pipeline.predict(x_test)
    dt_acc = accuracy_score(y_test, dt_pred)
    logger.print(f"Accuracy (DT): {dt_acc:.4f}")
    logger.print(classification_report(y_test, dt_pred, target_names=["Female", "Male"]))
    logger.print("")

    logger.print("=== Classic ML: SVM (RBF) ===")
    svm_pipeline.fit(x_train, y_train)
    svm_pred = svm_pipeline.predict(x_test)
    svm_acc = accuracy_score(y_test, svm_pred)
    logger.print(f"Accuracy (SVM): {svm_acc:.4f}")
    logger.print(classification_report(y_test, svm_pred, target_names=["Female", "Male"]))
    logger.print("")

    logger.print("=== Neural network: MLP (TensorFlow/Keras) ===")

    preprocessor_for_nn = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop",
    )

    x_train_nn = preprocessor_for_nn.fit_transform(x_train)
    x_test_nn = preprocessor_for_nn.transform(x_test)

    if hasattr(x_train_nn, "toarray"):
        x_train_nn = x_train_nn.toarray()
    if hasattr(x_test_nn, "toarray"):
        x_test_nn = x_test_nn.toarray()

    model = build_mlp(
        input_dim=x_train_nn.shape[1],
        hidden_units=args.hidden_units,
        hidden_layers=args.hidden_layers,
        dropout=args.dropout,
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=6,
        mode="max",
        restore_best_weights=True,
    )

    history = model.fit(
        x_train_nn,
        y_train,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=0,
        callbacks=[early_stop],
    )

    best_val_acc = max(history.history.get("val_accuracy", [0.0]))
    logger.print(f"Training finished. Best val_accuracy: {best_val_acc:.4f}")
    logger.print("")

    y_prob = model.predict(x_test_nn, verbose=0).ravel()
    y_pred_nn = (y_prob >= 0.5).astype(int)

    nn_acc = accuracy_score(y_test, y_pred_nn)
    logger.print(f"Accuracy (MLP): {nn_acc:.4f}")
    logger.print(classification_report(y_test, y_pred_nn, target_names=["Female", "Male"]))
    logger.print("")

    cm = confusion_matrix(y_test, y_pred_nn)
    save_confusion_matrix_png(
        cm=cm,
        labels=["Female", "Male"],
        out_path=cm_path,
        title="Confusion Matrix (MLP) - cats_dataset.csv",
    )
    logger.print(f"Saved confusion matrix to: {cm_path}")
    logger.print("")

    logger.print("=== SUMMARY (Accuracy) ===")
    logger.print(f"Decision Tree: {dt_acc:.4f}")
    logger.print(f"SVM (RBF):     {svm_acc:.4f}")
    logger.print(f"MLP (Keras):   {nn_acc:.4f}")
    logger.print("")
    logger.print("Log saved to:", log_path)
    logger.print("Done")

    logger.close()


if __name__ == "__main__":
    main()