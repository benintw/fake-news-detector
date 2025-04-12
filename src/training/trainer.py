import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

from ..data.dataloader import create_dataloaders
from ..model.model import FakeNewsClassifier
from .evaluator import Evaluator


class Trainer:

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.device = torch.device("mps")
        print(f"Trainer is Using device: {self.device}")

        # Create model and move to device
        self.model = FakeNewsClassifier(config)
        self.model.to(self.device)

        # Setup evaluator
        self.evaluator = Evaluator(self.model, config, self.device)

        # Initialize embeddings and labels storage for visualization
        self.val_embeddings = None
        self.val_labels = None
        self.test_embeddings = None
        self.test_labels = None

        # Get dataloaders
        dataloaders = create_dataloaders(config)
        self.train_dataloader = dataloaders["train"]
        self.val_dataloader = dataloaders["val"]
        self.test_dataloader = dataloaders["test"]

        # Setup loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"])

        # Setup directories for saving models and results
        self.output_dir = Path(config.get("output_dir", "./outputs"))
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.plots_dir = self.output_dir / "plots"

        # Create directories if they don't exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Initialize wandb if enabled
        self.use_wandb = config.get("use_wandb", False)
        if self.use_wandb:
            try:
                wandb.login()
                self.wandb_run = wandb.init(
                    project=config.get("wandb_project", "fake-news-detector"),
                    entity=config.get(
                        "wandb_entity", "chen-ben968-benchen"
                    ),  # Updated to match config
                    name=config.get("wandb_run_name", None),
                    config=config,
                )
            except Exception as e:
                print(f"Warning: Could not initialize wandb: {e}")
                self.use_wandb = False

    def train_epoch(self) -> Tuple[float, float]:

        self.model.train()
        total_train_loss = 0.0
        total_train_acc = 0.0

        batch_progress = tqdm(self.train_dataloader, desc="Training")
        dataloader_len = len(self.train_dataloader)  # Number of batches

        for batch in batch_progress:
            embeddings, labels = batch
            batch_size = labels.size(0)  # Get current batch size

            embeddings, labels = embeddings.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(embeddings)
            preds = torch.argmax(logits, dim=-1)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            # Calculate batch accuracy
            batch_acc = (preds == labels).sum().item() / batch_size

            # Track batch loss and accuracy (per batch)
            total_train_loss += loss.item()
            total_train_acc += batch_acc

            # Update progress bar with current metrics
            batch_progress.set_postfix(
                {
                    "loss": loss.item(),
                    "acc": batch_acc,
                }
            )

        # Return average loss and accuracy per batch
        return total_train_loss / dataloader_len, total_train_acc / dataloader_len

    def validate_epoch(self) -> Tuple[float, float]:
        avg_val_loss, avg_val_acc = self.evaluator.evaluate(self.val_dataloader)
        return avg_val_loss, avg_val_acc

    def train(self, epochs: Optional[int] = None) -> Dict[str, List[float]]:
        """
        Train the model for the specified number of epochs.

        Args:
            epochs: Number of epochs to train for. If None, uses config["EPOCHS"]

        Returns:
            Dictionary containing training history (losses and accuracies)
        """
        num_epochs = epochs if epochs is not None else self.config.get("EPOCHS", 10)

        # Initialize tracking variables
        best_val_loss = float("inf")
        best_val_acc = 0.0
        train_losses = []
        train_acc = []
        val_losses = []
        val_acc = []
        patience_counter = 0
        early_stopping_patience = self.config.get("early_stopping_patience", 5)

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")

            # Train and validate for one epoch
            train_loss, train_accuracy = self.train_epoch()
            val_loss, val_accuracy = self.validate_epoch()

            # Store metrics for plotting
            train_losses.append(train_loss)
            train_acc.append(train_accuracy)
            val_losses.append(val_loss)
            val_acc.append(val_accuracy)

            # Log to wandb if enabled
            if self.use_wandb:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "train_accuracy": train_accuracy,
                        "val_loss": val_loss,
                        "val_accuracy": val_accuracy,
                    }
                )

            # Save model if validation loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_model("best_loss_model.pth", epoch, val_loss, val_accuracy)
                patience_counter = 0
            else:
                patience_counter += 1

            # Also save model if validation accuracy improves
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                self._save_model(
                    "best_accuracy_model.pth", epoch, val_loss, val_accuracy
                )

            # Print epoch results
            print(
                f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}"
            )
            print(
                f"  Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}"
            )

            # Early stopping check
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        # Save final model
        self._save_model("final_model.pth", num_epochs, val_loss, val_accuracy)

        # Plot training history
        self.plot_training_history(
            train_losses, val_losses, train_acc, val_acc, save_dir=str(self.plots_dir)
        )

        # Log final plots to wandb
        if self.use_wandb:
            wandb.log(
                {
                    "training_history": wandb.Image(
                        str(self.plots_dir / "training_history.png")
                    )
                }
            )

        # Return training history
        return {
            "train_losses": train_losses,
            "train_acc": train_acc,
            "val_losses": val_losses,
            "val_acc": val_acc,
        }

    def _save_model(
        self, filename: str, epoch: int, val_loss: float, val_accuracy: float
    ) -> None:
        """
        Save model checkpoint with metadata.

        Args:
            filename: Name of the file to save the model to
            epoch: Current epoch number
            val_loss: Validation loss
            val_accuracy: Validation accuracy
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "config": self.config,
        }

        torch.save(checkpoint, self.checkpoint_dir / filename)
        print(f"  Model saved to {self.checkpoint_dir / filename}")

    def evaluate_model(
        self, dataloader=None, load_best=True, model_path=None
    ) -> Dict[str, Any]:
        """
        Perform detailed evaluation of the model.

        Args:
            dataloader: Dataloader to evaluate on. If None, uses validation dataloader
            load_best: Whether to load the best model before evaluation
            model_path: Path to model to load. If None and load_best is True, loads best_accuracy_model.pth

        Returns:
            Dictionary with evaluation metrics
        """
        if load_best:
            if model_path is None:
                model_path = self.checkpoint_dir / "best_accuracy_model.pth"

            if os.path.exists(model_path):
                checkpoint = torch.load(model_path)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                print(f"Loaded model from {model_path}")
            else:
                print(f"No model found at {model_path}, using current model")

        # Use validation dataloader if none provided
        eval_dataloader = dataloader if dataloader is not None else self.val_dataloader

        # Collect embeddings and labels for visualization
        self._collect_embeddings(
            eval_dataloader, is_test=(dataloader == self.test_dataloader)
        )

        # Run detailed evaluation
        results = self.evaluator.detailed_evaluation(eval_dataloader)

        # Print summary metrics
        print("\nEvaluation Results:")
        print(f"  Loss: {results['loss']:.4f}")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")
        print(f"  F1 Score: {results['f1']:.4f}")

        # Print classification report
        print("\nClassification Report:")
        for class_name, metrics in results["classification_report"].items():
            if isinstance(metrics, dict):
                print(
                    f"  {class_name}:\n    Precision: {metrics['precision']:.4f}\n    Recall: {metrics['recall']:.4f}\n    F1-Score: {metrics['f1-score']:.4f}\n    Support: {metrics['support']}"
                )

        # Plot confusion matrix
        self.evaluator.plot_confusion_matrix(
            results["confusion_matrix"],
            save_path=str(self.plots_dir / "confusion_matrix.png"),
        )

        # Log to wandb if enabled
        if self.use_wandb:
            wandb.log(
                {
                    "test_loss": results["loss"],
                    "test_accuracy": results["accuracy"],
                    "test_precision": results["precision"],
                    "test_recall": results["recall"],
                    "test_f1": results["f1"],
                    "confusion_matrix": wandb.Image(
                        str(self.plots_dir / "confusion_matrix.png")
                    ),
                }
            )

        return results

    def test(self) -> Dict[str, Any]:
        """
        Test the model on the test set.

        Returns:
            Dictionary with test metrics
        """
        print("\nEvaluating on test set...")
        return self.evaluate_model(dataloader=self.test_dataloader, load_best=True)

    def _collect_embeddings(self, dataloader, is_test=False):
        """
        Collect embeddings and labels from a dataloader for visualization.

        Args:
            dataloader: The dataloader to collect embeddings from
            is_test: Whether this is the test dataloader
        """
        self.model.eval()
        all_embeddings = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                embeddings, labels = batch
                embeddings, labels = embeddings.to(self.device), labels.to(self.device)

                # Store embeddings and labels
                all_embeddings.append(embeddings.cpu())
                all_labels.append(labels.cpu())

        # Concatenate all batches
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Store in appropriate attributes
        if is_test:
            self.test_embeddings = all_embeddings
            self.test_labels = all_labels
        else:
            self.val_embeddings = all_embeddings
            self.val_labels = all_labels

    def plot_training_history(
        self,
        train_losses: List[float],
        val_losses: List[float],
        train_acc: List[float],
        val_acc: List[float],
        save_dir: str = None,
    ) -> None:
        """
        Plot training and validation loss and accuracy.

        Args:
            train_losses: List of training losses per epoch
            val_losses: List of validation losses per epoch
            train_acc: List of training accuracies per epoch
            val_acc: List of validation accuracies per epoch
            save_dir: Directory to save plots, if None, just displays them
        """
        import matplotlib.pyplot as plt

        epochs = range(1, len(train_losses) + 1)

        # Plot losses
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, "b-", label="Training Loss")
        plt.plot(epochs, val_losses, "r-", label="Validation Loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        # Plot accuracies
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_acc, "b-", label="Training Accuracy")
        plt.plot(epochs, val_acc, "r-", label="Validation Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.tight_layout()

        if save_dir:
            plt.savefig(f"{save_dir}/training_history.png")
            plt.close()
        else:
            plt.show()
