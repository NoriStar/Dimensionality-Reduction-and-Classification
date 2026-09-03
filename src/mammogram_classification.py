"""Compare three educational baselines on BreastMNIST."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "breastmnist.npz"
RESULTS_DIR = ROOT / "results"
RANDOM_STATE = 42


def load_split(data: np.lib.npyio.NpzFile, split: str) -> tuple[np.ndarray, np.ndarray]:
    images = data[f"{split}_images"].reshape(len(data[f"{split}_images"]), -1).astype(float) / 255.0
    labels = data[f"{split}_labels"].reshape(-1).astype(int)
    return images, labels


class LeastSquaresClassifier:
    """Binary linear classifier fitted with a stable least-squares solution."""

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "LeastSquaresClassifier":
        design = np.column_stack([features, np.ones(len(features))])
        self.weights = np.linalg.lstsq(design, labels, rcond=None)[0]
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        design = np.column_stack([features, np.ones(len(features))])
        return (design @ self.weights >= 0.5).astype(int)


def evaluate(name: str, labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | str | int]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", pos_label=1, zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "model": name,
        "accuracy": accuracy_score(labels, predictions),
        "precision_malignant": precision,
        "recall_malignant": recall,
        "f1_malignant": f1,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def plot_mlp_training(model: MLPClassifier) -> None:
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.plot(model.loss_curve_, color="royalblue")
    axis.set(title="Neural-network training loss", xlabel="Iteration", ylabel="Cross-entropy loss")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "neural_network_loss.png", dpi=200, bbox_inches="tight")


def main() -> None:
    data = np.load(DATA_PATH)
    train_x, train_y = load_split(data, "train")
    test_x, test_y = load_split(data, "test")

    models = {
        "Least squares": LeastSquaresClassifier(),
        "Polynomial SVM": make_pipeline(StandardScaler(), SVC(kernel="poly", degree=3, C=1.0)),
        "Neural network": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True, random_state=RANDOM_STATE),
        ),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, model in models.items():
        model.fit(train_x, train_y)
        rows.append(evaluate(name, test_y, model.predict(test_x)))

    results = pd.DataFrame(rows).set_index("model")
    results.to_csv(RESULTS_DIR / "classification_metrics.csv")
    print(results.round(4).to_string())

    mlp = models["Neural network"].named_steps["mlpclassifier"]
    plot_mlp_training(mlp)


if __name__ == "__main__":
    main()
