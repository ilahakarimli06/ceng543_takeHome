"""
Visualize latent representations of trained sentiment models using PCA and t-SNE.

This script loads saved checkpoints from training, extracts hidden representations
on the IMDb test split, and produces 2D projections to highlight structural patterns.
"""

import argparse
import json
from pathlib import Path
import torch

from data_loader import get_dataloaders
from model import BiGRU, BiLSTM, BERTBiGRU, BERTBiLSTM
from utils import (
    analyze_separability,
    extract_features,
    plot_2d_projection,
    plot_combined_projection,
    project_2d,
    set_plot_style,
    set_seed,
)


MODEL_REGISTRY = {
    "glove_bilstm": {
        "title": "GloVe + BiLSTM",
        "checkpoint": "glove__bilstm_model.pt",
        "mode": "static",
        "class": BiLSTM,
    },
    "glove_bigru": {
        "title": "GloVe + BiGRU",
        "checkpoint": "glove__bigru_model.pt",
        "mode": "static",
        "class": BiGRU,
    },
    "bert_bilstm": {
        "title": "BERT + BiLSTM",
        "checkpoint": "bert__bilstm_model.pt",
        "mode": "contextual",
        "class": BERTBiLSTM,
    },
    "bert_bigru": {
        "title": "BERT + BiGRU",
        "checkpoint": "bert__bigru_model.pt",
        "mode": "contextual",
        "class": BERTBiGRU,
    },
}

DEFAULT_LIMIT_TRAIN = 5000
DEFAULT_LIMIT_TEST = 2000
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LEN = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize latent representations with PCA and t-SNE."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=list(MODEL_REGISTRY.keys()),
        help="Optional model key to visualize (default: all available checkpoints).",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path("."),
        help="Directory containing saved model checkpoints.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("question1/visualization"),
        help="Directory to store generated plots and metrics.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=2000,
        help="Maximum number of samples to use for visualization.",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity value (will be clipped if larger than sample count).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable tqdm progress bars during feature extraction.",
    )
    return parser.parse_args()


def infer_model_dims(state_dict: dict) -> tuple[int, int]:
    """Infer hidden dimension and number of classes from a checkpoint."""
    if "fc.weight" not in state_dict:
        raise KeyError("Checkpoint is missing 'fc.weight'; cannot infer model dimensions.")
    fc_weight = state_dict["fc.weight"]
    num_classes = fc_weight.shape[0]
    hidden_dim = fc_weight.shape[1] // 2
    return int(hidden_dim), int(num_classes)


def build_model(config, meta, state_dict: dict):
    """Instantiate a model with metadata-derived constructor arguments."""
    model_cls = config["class"]
    hidden_dim, num_classes = infer_model_dims(state_dict)
    if config["mode"] == "static":
        embedding_dim = meta["embedding_dim"]
        return model_cls(embedding_dim, hidden_dim, num_classes)
    bert_model_name = meta["model_name"]
    return model_cls(bert_model_name, hidden_dim, num_classes)


def ensure_checkpoint(path: Path) -> bool:
    if path.exists():
        return True
    print(f"Skipping - checkpoint not found: {path}")
    return False


def run_visualization(args: argparse.Namespace):
    set_seed(args.seed)
    set_plot_style()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target_models = (
        [args.model] if args.model else list(MODEL_REGISTRY.keys())
    )

    dataloader_cache = {}
    metrics_summary = {}

    print("\n" + "=" * 70)
    print("LATENT REPRESENTATION VISUALIZATION")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Output directory: {args.output_dir.resolve()}")
    print(f"Max samples per model: {args.max_samples}")

    for key in target_models:
        config = MODEL_REGISTRY[key]
        checkpoint_path = args.checkpoint_dir / config["checkpoint"]

        print("\n" + "-" * 70)
        print(f"Model: {config['title']}")
        print(f"Checkpoint: {checkpoint_path}")

        if not ensure_checkpoint(checkpoint_path):
            continue

        mode = config["mode"]
        if mode not in dataloader_cache:
            print("Preparing IMDb dataloader...")
            (
                _train_loader,
                _val_loader,
                test_loader,
                meta,
            ) = get_dataloaders(
                mode=mode,
                batch_size=DEFAULT_BATCH_SIZE,
                max_len=DEFAULT_MAX_LEN,
                limit_train=DEFAULT_LIMIT_TRAIN,
                limit_test=DEFAULT_LIMIT_TEST,
            )
            dataloader_cache[mode] = (test_loader, meta)
        else:
            test_loader, meta = dataloader_cache[mode]

        state_dict = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        model = build_model(config, meta, state_dict)
        model.load_state_dict(state_dict)
        model.to(device)

        print("Extracting latent representations...")
        features, labels = extract_features(
            model,
            test_loader,
            device,
            mode=mode,
            max_samples=args.max_samples,
            show_progress=not args.no_progress,
            progress_desc="Batches",
        )

        sample_count = features.shape[0]
        print(f"✓ Collected {sample_count} feature vectors")

        pca_reduced, pca_model = project_2d(
            features,
            method="pca",
            random_state=args.seed,
            return_model=True,
        )

        tsne_reduced = None
        tsne_perplexity = None
        if sample_count >= 10:
            max_valid = max(1.0, sample_count - 1)
            tsne_perplexity = min(args.perplexity, max_valid)
            if sample_count > 5:
                tsne_perplexity = max(5.0, tsne_perplexity)
            tsne_reduced = project_2d(
                features,
                method="tsne",
                random_state=args.seed,
                perplexity=tsne_perplexity,
            )
        else:
            print(" Not enough samples for reliable t-SNE (need >=10). Skipping t-SNE plots.")

        metrics = analyze_separability(features, labels)
        metrics_summary[key] = {
            "model_name": config["title"],
            "checkpoint": str(checkpoint_path.resolve()),
            "num_samples": int(sample_count),
            "separability": metrics,
            "pca_explained_variance": [
                float(x) for x in pca_model.explained_variance_ratio_
            ],
            "tsne_perplexity": float(tsne_perplexity) if tsne_perplexity else None,
        }

        plot_2d_projection(
            pca_reduced,
            labels,
            "PCA",
            config["title"],
            args.output_dir / f"{key}_pca.png",
        )
        if tsne_reduced is not None:
            plot_2d_projection(
                tsne_reduced,
                labels,
                "t-SNE",
                config["title"],
                args.output_dir / f"{key}_tsne.png",
            )
            plot_combined_projection(
                pca_reduced,
                tsne_reduced,
                labels,
                config["title"],
                args.output_dir / f"{key}_combined.png",
            )
            print(f"✓ Saved PCA, t-SNE, and combined figures to {args.output_dir}")
        else:
            print(f"✓ Saved PCA figure to {args.output_dir} (t-SNE skipped)")

    if metrics_summary:
        metrics_path = args.output_dir / "latent_visualization_metrics.json"
        with metrics_path.open("w") as fp:
            json.dump(metrics_summary, fp, indent=2)
        print(f"\nMetrics written to: {metrics_path.resolve()}")
    else:
        print("\nNo visualizations were generated (missing checkpoints?).")

    print("\nVisualization completed.")


def main():
    args = parse_args()
    run_visualization(args)


if __name__ == "__main__":
    main()

