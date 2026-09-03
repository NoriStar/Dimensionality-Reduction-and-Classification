"""Visualize high-dimensional BERT review embeddings in two dimensions."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "text_feature_vectors.npy"
OUTPUT_PATH = ROOT / "results" / "sentiment_embeddings.png"
RANDOM_STATE = 42


def load_features(path: Path) -> np.ndarray:
    features = np.load(path)
    if features.ndim != 2:
        raise ValueError(f"Expected a 2D array, received shape {features.shape}.")
    return features


def create_embeddings(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    standardized = StandardScaler().fit_transform(features)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_embedding = pca.fit_transform(standardized)

    laplacian = SpectralEmbedding(
        n_components=2,
        n_neighbors=10,
        affinity="nearest_neighbors",
        random_state=RANDOM_STATE,
    )
    laplacian_embedding = laplacian.fit_transform(standardized)
    explained_variance = float(pca.explained_variance_ratio_.sum())
    return pca_embedding, laplacian_embedding, explained_variance


def plot_embeddings(pca_data: np.ndarray, laplacian_data: np.ndarray) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(pca_data[:, 0], pca_data[:, 1], alpha=0.65, s=28)
    axes[0].set(title="PCA", xlabel="Principal component 1", ylabel="Principal component 2")

    axes[1].scatter(laplacian_data[:, 0], laplacian_data[:, 1], alpha=0.65, s=28, color="darkorange")
    axes[1].set(title="Laplacian Eigenmaps", xlabel="Embedding dimension 1", ylabel="Embedding dimension 2")

    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")


def main() -> None:
    features = load_features(DATA_PATH)
    pca_data, laplacian_data, explained_variance = create_embeddings(features)
    plot_embeddings(pca_data, laplacian_data)
    print(f"Samples: {features.shape[0]}, original features: {features.shape[1]}")
    print(f"Variance explained by PC1 and PC2: {explained_variance:.2%}")
    print(f"Saved figure to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
