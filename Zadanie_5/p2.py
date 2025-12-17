"""
Neural network Project 2
===================================

Authors:
- Wiktor Rapacz
- Hanna Paczoska

Problem statement:
Train a neural network to recognize animals in images using the CIFAR-10 dataset.
Use only the animal classes (bird, cat, deer, dog, frog, horse) and compare two CNN sizes.

Solution overview:
- Dataset: CIFAR-10 (loaded via tf.keras.datasets.cifar10)
- Subset: keep only animal classes (6 classes), remap labels to 0..5
- Models: two CNN architectures
  - "small" CNN
  - "large" CNN
- Evaluation:
  - accuracy + classification report
  - confusion matrix saved as PNG for each model

Data source:
- CIFAR-10 dataset provided by TensorFlow/Keras:
  tf.keras.datasets.cifar10.load_data()

How it works (high-level):
1) Load CIFAR-10.
2) Filter to animal classes and remap labels.
3) Normalize images to [0, 1].
4) Train and evaluate:
   - small CNN
   - large CNN
5) Save:
   - log: outputs/logs/p2_log.txt
   - confusion matrices:
     outputs/plots/p2_confusion_matrix_small.png
     outputs/plots/p2_confusion_matrix_large.png

Environment setup:
1) Use Python 3.10.x (recommended for TensorFlow 2.10.* on Windows).
2) Create/activate venv:
   - python -m venv .venv
   - .venv\\Scripts\\activate
3) Install dependencies:
   - pip install -r requirements.txt
4) Run:
   - python p2.py

Notes:
- Neural networks are implemented using a single framework: TensorFlow (tf.keras).
- Logs and plots are saved to the outputs/ directory.
"""

import os
import sys
import argparse
import random
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import tensorflow as tf
from tensorflow.keras import layers, models


@dataclass
class TeeLogger:
    """
    A small logger that writes messages both to console and to a text file.

    This helps to provide reproducible "evidence" (logs) for the repository
    as required by the course (screenshots/log files).
    """

    path: str

    def __post_init__(self) -> None:
        """Create parent directory and open the log file."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._f = open(self.path, "w", encoding="utf-8")

    def print(self, *args) -> None:
        """
        Print to console and append the same line to the log file.

        Args:
            *args: Objects to be printed (joined by spaces).
        """
        text = " ".join(str(a) for a in args)
        sys.stdout.write(text + "\n")
        self._f.write(text + "\n")
        self._f.flush()

    def close(self) -> None:
        """Close the log file safely."""
        try:
            self._f.close()
        except (OSError, ValueError):
            pass


def set_seeds(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Integer seed for random, NumPy and TensorFlow.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def save_confusion_matrix_png(cm: np.ndarray, labels: list[str], out_path: str, title: str) -> None:
    """
    Save a confusion matrix figure to a PNG file.

    Args:
        cm: Confusion matrix array of shape (n_classes, n_classes).
        labels: Class names used on the axes.
        out_path: Path to save the PNG file.
        title: Figure title.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)

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


# CIFAR-10 classes:
# 0 airplane, 1 automobile, 2 bird, 3 cat, 4 deer, 5 dog, 6 frog, 7 horse, 8 ship, 9 truck
ANIMAL_IDS = [2, 3, 4, 5, 6, 7]  # bird, cat, deer, dog, frog, horse
ANIMAL_LABELS = ["bird", "cat", "deer", "dog", "frog", "horse"]


def filter_animals(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Filter CIFAR-10 arrays to keep only animal classes and remap labels to 0..5.

    Args:
        x: Images array, shape (N, 32, 32, 3).
        y: Labels array, shape (N, 1) or (N,).

    Returns:
        (x_animals, y_remapped):
            x_animals: filtered images
            y_remapped: remapped labels in range [0..5]
    """
    y_flat = y.reshape(-1)
    mask = np.isin(y_flat, ANIMAL_IDS)
    x_a = x[mask]
    y_a = y_flat[mask]

    mapping = {cid: i for i, cid in enumerate(ANIMAL_IDS)}
    y_remap = np.array([mapping[int(v)] for v in y_a], dtype=np.int64)
    return x_a, y_remap


def build_cnn(model_size: str, input_shape: tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    """
    Build and compile a CNN for multi-class classification.

    Args:
        model_size: "small" or "large" architecture variant.
        input_shape: Input image shape (H, W, C), e.g. (32, 32, 3).
        num_classes: Number of output classes.

    Returns:
        A compiled tf.keras.Model.
    """
    if model_size not in {"small", "large"}:
        raise ValueError("model_size must be 'small' or 'large'")

    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))

    if model_size == "small":
        model.add(layers.Conv2D(32, 3, activation="relu", padding="same"))
        model.add(layers.MaxPooling2D())
        model.add(layers.Conv2D(64, 3, activation="relu", padding="same"))
        model.add(layers.MaxPooling2D())
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation="relu"))
        model.add(layers.Dropout(0.3))
    else:
        model.add(layers.Conv2D(32, 3, activation="relu", padding="same"))
        model.add(layers.Conv2D(32, 3, activation="relu", padding="same"))
        model.add(layers.MaxPooling2D())
        model.add(layers.Dropout(0.25))

        model.add(layers.Conv2D(64, 3, activation="relu", padding="same"))
        model.add(layers.Conv2D(64, 3, activation="relu", padding="same"))
        model.add(layers.MaxPooling2D())
        model.add(layers.Dropout(0.25))

        model.add(layers.Conv2D(128, 3, activation="relu", padding="same"))
        model.add(layers.MaxPooling2D())
        model.add(layers.Dropout(0.25))

        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation="relu"))
        model.add(layers.Dropout(0.4))

    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_and_eval(
    logger: TeeLogger,
    model_size: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    batch_size: int,
    seed: int,
    out_cm_path: str,
) -> float:
    """
    Train a CNN of a given size and evaluate it on a test set.

    Saves a confusion matrix plot to a PNG file and returns accuracy.

    Args:
        logger: TeeLogger instance.
        model_size: "small" or "large".
        x_train: Training images, normalized to [0,1].
        y_train: Training labels (0..5).
        x_test: Test images, normalized to [0,1].
        y_test: Test labels (0..5).
        epochs: Maximum number of epochs.
        batch_size: Batch size.
        seed: Seed for reproducibility.
        out_cm_path: Output path of the confusion matrix PNG.

    Returns:
        Test accuracy for this model size.
    """
    set_seeds(seed)

    # Make input shape explicit to keep IDE type checks happy (H, W, C)
    input_shape: tuple[int, int, int] = (
        int(x_train.shape[1]),
        int(x_train.shape[2]),
        int(x_train.shape[3]),
    )

    model = build_cnn(
        model_size=model_size,
        input_shape=input_shape,
        num_classes=len(ANIMAL_LABELS),
    )

    logger.print(f"\n=== Training CNN ({model_size}) ===")
    logger.print(model.summary())

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=4,
        mode="max",
        restore_best_weights=True,
    )

    model.fit(
        x_train,
        y_train,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[early_stop],
    )

    y_prob = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_test, y_pred)
    logger.print(f"Accuracy ({model_size}): {acc:.4f}")
    logger.print(classification_report(y_test, y_pred, target_names=ANIMAL_LABELS))

    cm = confusion_matrix(y_test, y_pred)
    save_confusion_matrix_png(
        cm=cm,
        labels=ANIMAL_LABELS,
        out_path=out_cm_path,
        title=f"Confusion Matrix (CNN {model_size}) - CIFAR-10 animals",
    )
    logger.print(f"Saved confusion matrix to: {out_cm_path}")

    return acc


def main() -> None:
    """
    Entry point.

    Loads CIFAR-10, keeps only animal classes, trains and evaluates two CNN sizes,
    and saves logs + confusion matrices.

    Outputs:
        - outputs/logs/p2_log.txt
        - outputs/plots/p2_confusion_matrix_small.png
        - outputs/plots/p2_confusion_matrix_large.png
    """
    parser = argparse.ArgumentParser(description="NAI - Task 5 - Step 2 (CIFAR-10 animals) - tf.keras")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs("outputs/logs", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)

    log_path = "outputs/logs/p2_log.txt"
    logger = TeeLogger(log_path)

    logger.print("=== Step 2: CIFAR-10 animals only | TensorFlow/Keras (tf.keras) ===")
    logger.print("Animal classes:", ANIMAL_LABELS)
    logger.print(f"epochs={args.epochs}, batch_size={args.batch_size}, seed={args.seed}")

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, y_train = filter_animals(x_train, y_train)
    x_test, y_test = filter_animals(x_test, y_test)

    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    logger.print(f"Train size (animals): {x_train.shape[0]}")
    logger.print(f"Test size  (animals): {x_test.shape[0]}")

    acc_small = train_and_eval(
        logger=logger,
        model_size="small",
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        out_cm_path="outputs/plots/p2_confusion_matrix_small.png",
    )

    acc_large = train_and_eval(
        logger=logger,
        model_size="large",
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        out_cm_path="outputs/plots/p2_confusion_matrix_large.png",
    )

    logger.print("\n=== SUMMARY (Accuracy) ===")
    logger.print(f"CNN small: {acc_small:.4f}")
    logger.print(f"CNN large: {acc_large:.4f}")
    logger.print("\nLog saved to:", log_path)
    logger.print("Done")

    logger.close()


if __name__ == "__main__":
    main()