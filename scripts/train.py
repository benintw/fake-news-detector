import json
from pathlib import Path

from src.training.trainer import Trainer
from src.utils.config import load_configs


def main() -> None:
    # Load and combine configurations
    config = load_configs(
        ["configs/model.yaml", "configs/training.yaml", "configs/dataset.yaml"]
    )

    # Create output directories if they don't exist
    output_dir = Path(config.get("output_dir", "./outputs"))
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create trainer and run
    trainer = Trainer(config)
    history = trainer.train()

    # Save training history
    with open(results_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=4)

    # Evaluate the model
    print("\nPerforming detailed evaluation on validation set...")
    val_results = trainer.evaluate_model(load_best=True)

    # Save validation results
    with open(results_dir / "validation_results.json", "w") as f:
        # Convert non-serializable values to strings
        serializable_results = {}
        for key, value in val_results.items():
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
        json.dump(serializable_results, f, indent=4)

    # Evaluate on test set if specified in config
    if config.get("eval_on_test", False):
        print("\nPerforming detailed evaluation on test set...")
        test_results = trainer.test()

        # Save test results
        with open(results_dir / "test_results.json", "w") as f:
            # Convert non-serializable values to strings
            serializable_results = {}
            for key, value in test_results.items():
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
            json.dump(serializable_results, f, indent=4)

    print(f"Training and evaluation complete. Results saved to {results_dir}")


if __name__ == "__main__":
    main()
