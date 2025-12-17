"""
Neural network Project 3
===================================

Authors:
- Wiktor Rapacz
- Hanna Paczoska

Problem statement:
Train a neural network to recognize clothing items (10 classes) using the Fashion-MNIST dataset.

Solution overview:
- Dataset: Fashion-MNIST (loaded via tf.keras.datasets.fashion_mnist)
- Model: Convolutional Neural Network (CNN) implemented with TensorFlow/Keras (tf.keras)
- Evaluation:
  - test accuracy and classification report
  - confusion matrix saved as PNG
  - training curve (train vs validation accuracy) saved as PNG

Data source:
- Fashion-MNIST (Zalando Research), loaded automatically via:
  tf.keras.datasets.fashion_mnist.load_data()

How it works (high-level):
1) Load Fashion-MNIST.
2) Normalize images to [0, 1] and add the channel dimension (28x28x1).
3) Train a CNN with early stopping.
4) Evaluate on the test set and generate metrics.
5) Save:
   - outputs/logs/p3_log.txt
   - outputs/plots/p3_confusion_matrix.png
   - outputs/plots/p3_training_curve.png

Environment setup:
1) Use Python 3.10.x (recommended for TensorFlow 2.10.* on Windows).
2) Create/activate venv:
   - python -m venv .venv
   - .venv\\Scripts\\activate
3) Install dependencies:
   - pip install -r requirements.txt
4) Run:
   - python p3.py

Notes:
- Neural networks are implemented using a single framework: TensorFlow (tf.keras).
- Fashion-MNIST is downloaded automatically by TensorFlow/Keras.
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
    A small logger that writes messages to both console and a text file.

    This is useful for assignments where logs (evidence) must be included
    in the repository under outputs/logs/.
    """

    path: str

    def __post_init__(self) -> None:
        """Create the parent directory and open the log file."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._f = open(self.path, "w", encoding="utf-8")

    def print(self, *args) -> None:
        """
        Print to console and append the same line to the log file.

        Args:
            *args: Objects to be printed (joined by spaces).
        """
        txt = " ".join(str(a) for a in args)
        sys.stdout.write(txt + "\n")
        self._f.write(txt + "\n")
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
        seed: Integer seed for Python random, NumPy, and TensorFlow.
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
                f"{cm[i, j]:d}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


FASHION_MNIST_LABELS = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def build_cnn(input_shape: tuple[int, int, int], model_size: str) -> tf.keras.Model:
    """
    Build and compile a CNN model for Fashion-MNIST.

    Args:
        input_shape: Input image shape (H, W, C), e.g. (28, 28, 1).
        model_size: "small" or "large" architecture variant.

    Returns:
        A compiled tf.keras.Model.
    """
    if model_size not in {"small", "large"}:
        raise ValueError("model_size must be 'small' or 'large'")

    m = models.Sequential()
    m.add(layers.Input(shape=input_shape))

    if model_size == "small":
        m.add(layers.Conv2D(32, 3, activation="relu", padding="same"))
        m.add(layers.MaxPooling2D())
        m.add(layers.Conv2D(64, 3, activation="relu", padding="same"))
        m.add(layers.MaxPooling2D())
        m.add(layers.Flatten())
        m.add(layers.Dense(128, activation="relu"))
        m.add(layers.Dropout(0.3))
    else:
        m.add(layers.Conv2D(32, 3, activation="relu", padding="same"))
        m.add(layers.Conv2D(32, 3, activation="relu", padding="same"))
        m.add(layers.MaxPooling2D())
        m.add(layers.Dropout(0.25))

        m.add(layers.Conv2D(64, 3, activation="relu", padding="same"))
        m.add(layers.Conv2D(64, 3, activation="relu", padding="same"))
        m.add(layers.MaxPooling2D())
        m.add(layers.Dropout(0.25))

        m.add(layers.Flatten())
        m.add(layers.Dense(256, activation="relu"))
        m.add(layers.Dropout(0.4))

    m.add(layers.Dense(10, activation="softmax"))

    m.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return m


def main() -> None:
    """
    Entry point.

    Loads Fashion-MNIST, trains a CNN with early stopping, evaluates on the test set,
    and saves logs + confusion matrix + training curve plots.

    Outputs:
        - outputs/logs/p3_log.txt
        - outputs/plots/p3_confusion_matrix.png
        - outputs/plots/p3_training_curve.png
    """
    parser = argparse.ArgumentParser(description="Task 3 - Fashion-MNIST classification (tf.keras)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_size", choices=["small", "large"], default="small")
    args = parser.parse_args()

    set_seeds(args.seed)

    os.makedirs("outputs/logs", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)

    log_path = "outputs/logs/p3_log.txt"
    cm_path = "outputs/plots/p3_confusion_matrix.png"
    curve_path = "outputs/plots/p3_training_curve.png"

    logger = TeeLogger(log_path)
    logger.print("=== Task 3: Fashion-MNIST (10 classes) ===")
    logger.print("Framework: TensorFlow/Keras (tf.keras)")
    logger.print(f"Model size: {args.model_size}")
    logger.print(f"epochs={args.epochs}, batch_size={args.batch_size}, seed={args.seed}")
    logger.print("")

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # Normalize to [0, 1] and add channel dimension
    x_train = (x_train.astype(np.float32) / 255.0)[..., np.newaxis]
    x_test = (x_test.astype(np.float32) / 255.0)[..., np.newaxis]

    input_shape: tuple[int, int, int] = (
        int(x_train.shape[1]),
        int(x_train.shape[2]),
        int(x_train.shape[3]),
    )

    logger.print(f"Train size: {x_train.shape[0]}, Test size: {x_test.shape[0]}")
    logger.print(f"Input shape: {input_shape}")
    logger.print("")

    model = build_cnn(input_shape=input_shape, model_size=args.model_size)
    logger.print(model.summary())

    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        mode="max",
        restore_best_weights=True,
    )

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.15,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=0,
        callbacks=[early],
    )

    probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    logger.print(f"Accuracy (test): {acc:.4f}")
    logger.print(classification_report(y_test, y_pred, target_names=FASHION_MNIST_LABELS))
    logger.print("")

    cm = confusion_matrix(y_test, y_pred)
    save_confusion_matrix_png(cm, FASHION_MNIST_LABELS, cm_path, "Confusion Matrix - Fashion-MNIST")
    logger.print("Saved confusion matrix to:", cm_path)

    # Training curve
    plt.figure()
    plt.plot(history.history.get("accuracy", []), label="train_acc")
    plt.plot(history.history.get("val_accuracy", []), label="val_acc")
    plt.title("Training curve")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_path, dpi=200)
    plt.close()
    logger.print("Saved training curve to:", curve_path)

    logger.print("Log saved to:", log_path)
    logger.print("Done ✅")
    logger.close()


if __name__ == "__main__":
    main()