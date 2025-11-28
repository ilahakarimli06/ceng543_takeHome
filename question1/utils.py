# utils.py
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm


# -------------------------------
# 1) Reproducibility
# -------------------------------
def set_seed(seed: int = 42) -> None:
    """Fix all RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # (cuDNN determinism trades some speed for determinism)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------------
# 2) Convergence efficiency helper
# -------------------------------

class EpochTimer:
    """Context manager: measure wall-clock seconds per epoch."""
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.seconds = time.time() - self.start



# -------------------------------
# 3) Batch helpers (mode-agnostic)
# -------------------------------
def prepare_batch(batch, device: torch.device, mode: str):
    """
    Normalize batch to (model_inputs, labels) and move to device.
    Handles both static and contextual modes.
    """
    x, y = batch
    if mode == "static":
        # x = {"inputs": [B, L, D], "lengths": [B]}
        x = {k: v.to(device) for k, v in x.items()}
        y = y.to(device)
        return x, y
    else:
        # x = {"input_ids", "attention_mask"(, "token_type_ids")}
        x = {k: v.to(device) for k, v in x.items()}
        y = y.to(device)
        return x, y


# -------------------------------
# 4) Latent visualization (PCA / t-SNE)
#    - You pass hidden features [N, H] and labels [N]
#    - Returns 2D numpy array for plotting in your notebook/report
# -------------------------------
def project_2d(
    features: np.ndarray,
    method: str = "pca",
    random_state: int = 42,
    return_model: bool = False,
    **kwargs,
) -> np.ndarray:
    """
    Reduce high-D features to 2D for visualization.
    method: "pca" or "tsne"
    """
    method = method.lower()
    if method == "pca":
        reducer = PCA(n_components=2, random_state=random_state, **kwargs)
    elif method in ("tsne", "t-sne"):
        reducer = TSNE(
            n_components=2,
            random_state=random_state,
            init="random",
            **kwargs,
        )
    else:
        raise ValueError("method must be 'pca' or 'tsne'")

    projected = reducer.fit_transform(features)
    if return_model:
        return projected, reducer
    return projected


def extract_features(
    model,
    data_loader,
    device,
    mode: str = "static",
    max_samples: int = 1000,
    show_progress: bool = False,
    progress_desc: str = "Extracting",
):
    """
    Extract latent representations from a model without projecting.
    Iterates over the data loader, collects features and labels up to max_samples.

    """
    model = model.to(device)
    model.eval()

    all_features: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    total_samples = 0

    iterator = data_loader
    if show_progress:
        iterator = tqdm(data_loader, desc=progress_desc, unit="batch", leave=False)

    with torch.no_grad():
        for batch in iterator:
            inputs, labels = prepare_batch(batch, device, mode)
            features = model.get_features(inputs)

            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            batch_size = labels.size(0)
            total_samples += batch_size
            if total_samples >= max_samples:
                # Stop early once we’ve gathered enough samples
                break

    if not all_features:
        raise RuntimeError("No features were extracted; check the dataloader and model.")

    features_array = np.concatenate(all_features, axis=0)[:max_samples]
    labels_array = np.concatenate(all_labels, axis=0)[:max_samples]
    return features_array, labels_array


def extract_and_visualize_features(model, data_loader, device, mode="static", method="pca", max_samples=1000):
    
    #Extract features and project them to 2D for visualization.
    
    features, labels = extract_features(
        model,
        data_loader,
        device,
        mode=mode,
        max_samples=max_samples,
    )

    # Project to 2D
    features_2d = project_2d(features, method=method)

    return features_2d, labels


# -------------------------------
# 5) Simple history container for logging
# -------------------------------
@dataclass
class History:
    train_loss: List[float]
    val_acc: List[float]
    val_macro_f1: List[float]
    epoch_time_sec: List[float]

    def add(self, loss: float, acc: float, f1: float, tsec: float):
        self.train_loss.append(loss)
        self.val_acc.append(acc)
        self.val_macro_f1.append(f1)
        self.epoch_time_sec.append(tsec)

    def last(self) -> Dict[str, float]:
        if not self.train_loss:
            return {}
        return {
            "train_loss": self.train_loss[-1],
            "val_acc": self.val_acc[-1],
            "val_macro_f1": self.val_macro_f1[-1],
            "epoch_time_sec": self.epoch_time_sec[-1],
        }


# -------------------------------
# Visualization helpers
# -------------------------------
def set_plot_style():
    """
    Configure Matplotlib defaults for cleaner plots.
    Sets style and font sizes for publication-quality figures.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.figsize": (10, 8),
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "font.size": 11,
    })


def analyze_separability(representations: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Compute simple metrics describing class separability in latent space.
    Returns center distance, average within-class variance, and a separability score.
    """
    labels = np.asarray(labels)
    representations = np.asarray(representations)

    pos_repr = representations[labels == 1]
    neg_repr = representations[labels == 0]

    pos_center = np.mean(pos_repr, axis=0)
    neg_center = np.mean(neg_repr, axis=0)

    center_distance = float(np.linalg.norm(pos_center - neg_center))

    pos_var = float(np.mean(np.var(pos_repr, axis=0)))
    neg_var = float(np.mean(np.var(neg_repr, axis=0)))
    avg_var = float((pos_var + neg_var) / 2.0) if (pos_repr.size and neg_repr.size) else 0.0

    separability = center_distance / np.sqrt(avg_var) if avg_var > 0 else float("inf")

    return {
        "center_distance": center_distance,
        "avg_within_class_variance": avg_var,
        "separability_score": float(separability),
    }


def plot_2d_projection(
    reduced: np.ndarray,
    labels: np.ndarray,
    method_name: str,
    title: str,
    output_file,
):
    """Render and save a 2D projection scatter plot."""
    labels = np.asarray(labels)
    reduced = np.asarray(reduced)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    mask_pos = labels == 1
    mask_neg = labels == 0

    ax.scatter(
        reduced[mask_neg, 0],
        reduced[mask_neg, 1],
        c="tab:blue",
        label="Negative",
        alpha=0.55,
        s=22,
        edgecolors="none",
    )
    ax.scatter(
        reduced[mask_pos, 0],
        reduced[mask_pos, 1],
        c="tab:red",
        label="Positive",
        alpha=0.55,
        s=22,
        edgecolors="none",
    )

    ax.set_xlabel(f"{method_name} Component 1")
    ax.set_ylabel(f"{method_name} Component 2")
    ax.set_title(f"{method_name} Projection - {title}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_combined_projection(
    pca_reduced: np.ndarray,
    tsne_reduced: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_file,
):
    """
    Save a side-by-side comparison of PCA and t-SNE projections.
    Useful for comparing linear and nonlinear dimensionality reduction.
    """
    labels = np.asarray(labels)
    pca_reduced = np.asarray(pca_reduced)
    tsne_reduced = np.asarray(tsne_reduced)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    mask_pos = labels == 1
    mask_neg = labels == 0

    # PCA
    ax = axes[0]
    ax.scatter(
        pca_reduced[mask_neg, 0],
        pca_reduced[mask_neg, 1],
        c="tab:blue",
        label="Negative",
        alpha=0.55,
        s=22,
        edgecolors="none",
    )
    ax.scatter(
        pca_reduced[mask_pos, 0],
        pca_reduced[mask_pos, 1],
        c="tab:red",
        label="Positive",
        alpha=0.55,
        s=22,
        edgecolors="none",
    )
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("PCA Projection")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # t-SNE
    ax = axes[1]
    ax.scatter(
        tsne_reduced[mask_neg, 0],
        tsne_reduced[mask_neg, 1],
        c="tab:blue",
        label="Negative",
        alpha=0.55,
        s=22,
        edgecolors="none",
    )
    ax.scatter(
        tsne_reduced[mask_pos, 0],
        tsne_reduced[mask_pos, 1],
        c="tab:red",
        label="Positive",
        alpha=0.55,
        s=22,
        edgecolors="none",
    )
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.set_title("t-SNE Projection")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Latent Space Visualization - {title}", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
