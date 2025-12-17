"""
Neural network Project 4
===================================

Authors:
- Wiktor Rapacz
- Hanna Paczoska

Problem statement:
Propose and implement a custom neural-network classification use case.
This script performs fruit image classification.

Solution overview:
- Dataset: Fruits-360 "3-body-problem" (Apples / Cherries / Tomatoes)
- Model: Convolutional Neural Network (CNN) implemented with TensorFlow/Keras (tf.keras)
- Evaluation:
  - test accuracy and classification report
  - confusion matrix saved as PNG
  - training curve (train vs validation accuracy) saved as PNG
- Practical repository rule:
  - the dataset is not committed to GitHub; it is downloaded automatically by the script.

Data source:
- Fruits-360 "3-body-problem" repository (downloaded as a ZIP archive):
  https://github.com/fruits-360/fruits-360-3-body-problem/archive/refs/heads/main.zip

How it works (high-level):
1) Ensure output folders exist (outputs/logs, outputs/plots).
2) Download the dataset ZIP from GitHub (if not already downloaded) and extract it to:
   data/fruits360_3body/
3) Build TensorFlow datasets from folders:
   - Training/ (train/validation split done automatically)
   - Test/
4) Train a small CNN model.
5) Evaluate on the test set and save:
   - outputs/logs/p4_log.txt
   - outputs/plots/p4_confusion_matrix.png
   - outputs/plots/p4_training_curve.png

Environment setup:
1) Use Python 3.10.x (recommended for TensorFlow 2.10.* on Windows).
2) Create/activate venv:
   - python -m venv .venv
   - .venv\\Scripts\\activate
3) Install dependencies:
   - pip install -r requirements.txt
4) Run:
   - python p4.py

Notes:
- Neural networks are implemented using a single framework: TensorFlow (tf.keras).
"""

import argparse
import zipfile
from dataclasses import dataclass
from pathlib import Path
import urllib.request

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import tensorflow as tf
from tensorflow.keras import layers, models


@dataclass
class Cfg:
    """
    Configuration container for the fruit classification experiment.

    Attributes:
        seed: Random seed for reproducibility.
        img_size: Image resize target (img_size x img_size).
        batch_size: Batch size for training and evaluation.
        epochs: Number of training epochs.
        lr: Learning rate for Adam optimizer.

        data_dir: Base directory for datasets.
        outputs_dir: Base directory for outputs.
        logs_dir: Directory where log files are written.
        plots_dir: Directory where plots (PNG) are saved.

        dataset_root_name: Subfolder name under data_dir where this dataset is stored.
        url_main: GitHub ZIP URL (main branch).
        url_master: Fallback GitHub ZIP URL (master branch).
    """

    seed: int = 42
    img_size: int = 100
    batch_size: int = 32
    epochs: int = 6
    lr: float = 1e-3

    data_dir: Path = Path("data")
    outputs_dir: Path = Path("outputs")
    logs_dir: Path = Path("outputs/logs")
    plots_dir: Path = Path("outputs/plots")

    dataset_root_name: str = "fruits360_3body"

    url_main: str = "https://github.com/fruits-360/fruits-360-3-body-problem/archive/refs/heads/main.zip"
    url_master: str = "https://github.com/fruits-360/fruits-360-3-body-problem/archive/refs/heads/master.zip"


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducible results.

    Args:
        seed: Integer seed used by Python random, NumPy and TensorFlow.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dirs(cfg: Cfg) -> None:
    """
    Create required directories if they do not exist.

    Args:
        cfg: Experiment configuration.
    """
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    cfg.plots_dir.mkdir(parents=True, exist_ok=True)


def download_and_extract(cfg: Cfg) -> Path:
    """
    Download and extract the dataset if it is not present locally.

    The dataset is downloaded as a ZIP from GitHub and extracted into:
        data/<dataset_root_name>/

    After extraction, this function searches for a folder that contains:
        Training/ and Test/

    Args:
        cfg: Experiment configuration.

    Returns:
        Path to the extracted dataset root that contains Training/ and Test/.

    Raises:
        RuntimeError: If Training/Test directories cannot be found after extraction.
    """
    root = cfg.data_dir / cfg.dataset_root_name
    root.mkdir(parents=True, exist_ok=True)

    existing_training = list(root.rglob("Training"))
    existing_test = list(root.rglob("Test"))
    if existing_training and existing_test:
        return existing_training[0].parent

    zip_path = root / "fruits360_3body.zip"

    def _download(url: str) -> None:
        """
        Download a ZIP archive from the given URL.

        Args:
            url: Download URL for the ZIP archive.
        """
        print(f"Downloading dataset from: {url}")
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, zip_path.as_posix())

    try:
        _download(cfg.url_main)
    except (OSError, IOError, RuntimeError):
        _download(cfg.url_master)

    print(f"Extracting: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)

    extracted_training = list(root.rglob("Training"))
    extracted_test = list(root.rglob("Test"))
    if not extracted_training or not extracted_test:
        raise RuntimeError("After extraction, could not find Training/ and Test/ folders.")

    extracted_root = extracted_training[0].parent
    print(f"Dataset extracted to: {extracted_root}")
    return extracted_root


def make_model(num_classes: int, img_size: int, lr: float) -> tf.keras.Model:
    """
    Build and compile a CNN classifier.

    Args:
        num_classes: Number of classes.
        img_size: Input image size (img_size x img_size).
        lr: Learning rate for Adam optimizer.

    Returns:
        A compiled tf.keras.Model.
    """
    model = models.Sequential(
        [
            layers.Input(shape=(img_size, img_size, 3)),
            layers.Rescaling(1.0 / 255.0),

            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),

            layers.Conv2D(128, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),

            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_training_curve(history: tf.keras.callbacks.History, out_path: Path) -> None:
    """
    Save training vs validation accuracy curve.

    Args:
        history: Keras History object returned by model.fit().
        out_path: Path to save the plot PNG.
    """
    plt.figure()
    plt.plot(history.history.get("accuracy", []), label="train_acc")
    plt.plot(history.history.get("val_accuracy", []), label="val_acc")
    plt.title("Training curve")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out_path: Path) -> None:
    """
    Save a confusion matrix plot as a PNG file.

    Args:
        cm: Confusion matrix array of shape (n_classes, n_classes).
        class_names: Class names for axis labels.
        out_path: Path to save the plot PNG.
    """
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix - Fruits (3-body-problem)")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> int:
    """
    Entry point.

    Downloads/extracts the fruit dataset if needed, trains a CNN classifier,
    evaluates on the test set, and writes logs/plots to the outputs directory.

    Returns:
        Process exit code (0 for success).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = Cfg(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
    )

    ensure_dirs(cfg)
    set_seed(cfg.seed)

    log_path = cfg.logs_dir / "p4_log.txt"

    def log(msg: str) -> None:
        """
        Append a message to the log file and also print it to console.

        Args:
            msg: Log message.
        """
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    if log_path.exists():
        log_path.unlink()

    log("=== p4.py (Fruits classification) ===")
    log("Framework: TensorFlow / Keras (tf.keras)")
    log("Dataset: Fruits-360 '3-body-problem' (Apples / Cherries / Tomatoes)")
    log(f"seed={cfg.seed}, img_size={cfg.img_size}, batch_size={cfg.batch_size}, epochs={cfg.epochs}")

    extracted_root = download_and_extract(cfg)
    train_dir = extracted_root / "Training"
    test_dir = extracted_root / "Test"

    if not train_dir.exists() or not test_dir.exists():
        raise RuntimeError("Training/Test folders not found in extracted dataset.")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        image_size=(cfg.img_size, cfg.img_size),
        batch_size=cfg.batch_size,
        shuffle=True,
        seed=cfg.seed,
        validation_split=0.2,
        subset="training",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        image_size=(cfg.img_size, cfg.img_size),
        batch_size=cfg.batch_size,
        shuffle=True,
        seed=cfg.seed,
        validation_split=0.2,
        subset="validation",
    )

    class_names = list(train_ds.class_names)
    num_classes = len(class_names)
    log(f"Classes ({num_classes}): {class_names}")

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="int",
        image_size=(cfg.img_size, cfg.img_size),
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    model = make_model(num_classes=num_classes, img_size=cfg.img_size, lr=cfg.lr)
    log("\n=== Model summary ===")
    model.summary(print_fn=log)

    log("\n=== Training ===")
    history = model.fit(train_ds, validation_data=val_ds, epochs=cfg.epochs, verbose=1)

    log("\n=== Test evaluation ===")
    y_true: list[int] = []
    y_pred: list[int] = []

    for batch_x, batch_y in test_ds:
        probs = model.predict(batch_x, verbose=0)
        preds = np.argmax(probs, axis=1)
        y_true.extend(batch_y.numpy().tolist())
        y_pred.extend(preds.tolist())

    y_true_arr = np.array(y_true, dtype=int)
    y_pred_arr = np.array(y_pred, dtype=int)

    acc = accuracy_score(y_true_arr, y_pred_arr)
    log(f"Test accuracy: {acc:.4f}")

    cm = confusion_matrix(y_true_arr, y_pred_arr)
    log("\nConfusion matrix:\n" + str(cm))
    log("\nClassification report:\n" + classification_report(y_true_arr, y_pred_arr, target_names=class_names))

    cm_path = cfg.plots_dir / "p4_confusion_matrix.png"
    curve_path = cfg.plots_dir / "p4_training_curve.png"

    plot_confusion_matrix(cm, class_names, cm_path)
    plot_training_curve(history, curve_path)

    log(f"\nSaved confusion matrix to: {cm_path.as_posix()}")
    log(f"Saved training curve to: {curve_path.as_posix()}")
    log(f"\nLog saved to: {log_path.as_posix()}")
    log("Done")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())