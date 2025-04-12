import argparse
import json
import os
from pathlib import Path

import torch
from src.training.trainer import Trainer
from src.utils.config import load_configs
from src.visualization.embedding_visualizer import EmbeddingVisualizer


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate fake news detection model")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model checkpoint to evaluate",
    )
    parser.add_argument(
        "--best", action="store_true", help="Use best model based on accuracy"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Evaluate on test set instead of validation set",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=True,
        help="Generate embedding visualizations",
    )
    parser.add_argument(
        "--vis_methods",
        type=str,
        default="pca,tsne",
        help="Visualization methods to use (comma-separated)",
    )
    args = parser.parse_args()

    # Load configurations
    config = load_configs(
        ["configs/model.yaml", "configs/training.yaml", "configs/dataset.yaml"]
    )

    # Create output directories
    output_dir = Path(config.get("output_dir", "./outputs"))
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Initialize trainer
    trainer = Trainer(config)

    # Load best model if specified
    if args.best or args.model_path:
        model_path = args.model_path or str(
            Path(config.get("checkpoints_dir", "./outputs/checkpoints"))
            / "best_accuracy_model.pth"
        )
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}...")
            checkpoint = torch.load(model_path)
            trainer.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Model loaded successfully from {model_path}")
        else:
            print(f"Warning: Model file {model_path} not found. Using default model.")

    # Evaluate model
    if args.test:
        # Evaluate on test set
        print("Evaluating on test set...")
        results = trainer.evaluator.detailed_evaluation(trainer.test_dataloader)
        output_file = results_dir / "test_results.json"
    else:
        # Evaluate on validation set
        print("Evaluating on validation set...")
        results = trainer.evaluator.detailed_evaluation(trainer.val_dataloader)
        output_file = results_dir / "validation_results.json"

    # Save serializable results
    serializable_results = {}
    for key, value in results.items():
        if key in ["loss", "accuracy", "precision", "recall", "f1"]:
            serializable_results[key] = float(value)
        elif key == "classification_report":
            serializable_results[key] = {}
            for class_name, metrics in value.items():
                if isinstance(metrics, dict):
                    serializable_results[key][class_name] = {
                        k: float(v) if isinstance(v, (int, float)) else v
                        for k, v in metrics.items()
                    }

    # Save results
    with open(output_file, "w") as f:
        json.dump(serializable_results, f, indent=4)

    # Save confusion matrix visualization
    if "confusion_matrix" in results:
        # Create plots directory if it doesn't exist
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        cm_path = (
            plots_dir / f"{'test' if args.test else 'validation'}_confusion_matrix.png"
        )
        print(f"Saving confusion matrix to {cm_path}...")
        trainer.evaluator.plot_confusion_matrix(
            results["confusion_matrix"], str(cm_path)
        )

    print(f"Evaluation complete. Results saved to {output_file}")

    # Generate embedding visualizations if requested
    if args.visualize:
        print("Generating embedding visualizations...")

        # Create visualizations directory
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

        # Collect embeddings and labels for visualization
        print("Collecting embeddings for visualization...")
        trainer._collect_embeddings(
            trainer.test_dataloader if args.test else trainer.val_dataloader,
            is_test=args.test,
        )

        # Get embeddings and labels
        if args.test:
            embeddings = trainer.test_embeddings
            labels = trainer.test_labels
            dataset_name = "test"
        else:
            embeddings = trainer.val_embeddings
            labels = trainer.val_labels
            dataset_name = "validation"

        # Convert embeddings to numpy if they're torch tensors
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        # Initialize visualizer
        config["use_wandb"] = config.get("use_wandb", True)  # Enable wandb logging
        visualizer = EmbeddingVisualizer(config)

        # Parse visualization methods
        methods = args.vis_methods.split(",")

        # Generate visualizations
        save_dir = vis_dir / dataset_name
        save_dir.mkdir(parents=True, exist_ok=True)

        results = visualizer.visualize_embeddings(
            embeddings=embeddings,
            labels=labels,
            methods=methods,
            save_dir=str(save_dir),
        )

        print(f"Visualizations saved to {save_dir}")


if __name__ == "__main__":
    main()
