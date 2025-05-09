import json

import click
from colorama import Fore, Style, init

from src.inference.predictor import NewsPredictor
from src.utils.config import load_configs

# Initialize colorama for cross-platform colored terminal output
init()


@click.command()
@click.option("--text", default=None, help="Text of a news article to classify")
@click.option(
    "--file",
    "file_path",
    default="./sample_news.txt",
    help="Path to a file with news articles to classify",
)
@click.option(
    "--model-path",
    default="./outputs/checkpoints/best_accuracy_model.pth",
    help="Path to model checkpoint to use",
)
@click.option(
    "--output", default="predictions.json", help="Path to save prediction results"
)
@click.option("--verbose", is_flag=True, help="Show detailed prediction information")
def main(text: str, file_path: str, model_path: str, output: str, verbose: bool):
    """Predict if news articles are real or fake."""
    # Load configurations
    config = load_configs(
        ["configs/model.yaml", "configs/training.yaml", "configs/dataset.yaml"]
    )

    # Initialize predictor
    predictor = NewsPredictor(config, model_path=model_path)

    # Make predictions
    if text:
        # Predict single text
        results = predictor.predict_text(text)
        _display_prediction_results([results], verbose)
    elif file_path:
        # Predict from file
        try:
            results = predictor.predict_file(file_path)
            print(
                f"\nAnalyzed {len(results)} article{'s' if len(results) > 1 else ''} from {file_path}"
            )
            _display_prediction_results(results, verbose)
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            return
        except Exception as e:
            print(f"Error processing file: {e}")
            return
    else:
        raise click.UsageError("Either --text or --file must be provided")

    # Save results
    with open(output, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nPredictions saved to {output}")


def _display_prediction_results(results, verbose=False):
    """Display prediction results in a user-friendly format.

    Args:
        results: List of prediction result dictionaries
        verbose: Whether to show detailed information
    """
    for i, result in enumerate(results):
        if len(results) > 1:
            print(f"\nArticle {i+1}:")

        # Determine color based on prediction
        if result["prediction"] == "real":
            color = Fore.GREEN
            emoji = "✅"
        else:  # fake
            color = Fore.RED
            emoji = "❌"

        # Display prediction with color
        confidence = result["confidence"] * 100
        print(
            f"{color}{emoji} Prediction: {result['prediction'].upper()} "
            f"(Confidence: {confidence:.1f}%){Style.RESET_ALL}"
        )

        # Show detailed probabilities if verbose
        if verbose:
            print("  Class probabilities:")
            for class_name, prob in result["probabilities"].items():
                class_color = Fore.GREEN if class_name == "real" else Fore.RED
                print(
                    f"    {class_color}{class_name}: {prob*100:.1f}%{Style.RESET_ALL}"
                )


if __name__ == "__main__":
    main()
