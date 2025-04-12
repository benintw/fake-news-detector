from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tqdm import tqdm


class Evaluator:

    def __init__(self, model, config: dict[str, Any], device: torch.device) -> None:
        self.model = model
        self.config = config
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

        # Get class mapping from config
        self.class_mapping = config.get("class_mapping", {0: "real", 1: "fake"})

    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """
        Evaluate the model on the given dataloader.

        Returns:
            float: The loss of the model on the dataloader
            float: The accuracy of the model on the dataloader
        """
        self.model.eval()

        total_loss = 0.0
        total_acc = 0.0
        all_preds = []
        all_labels = []

        batch_progress = tqdm(dataloader, desc="Evaluating")
        dataloader_len = len(dataloader)  # Number of batches

        for batch in batch_progress:
            embeddings, labels = batch
            batch_size = labels.size(0)  # Get current batch size

            embeddings, labels = embeddings.to(self.device), labels.to(self.device)

            with torch.inference_mode():
                logits = self.model(embeddings)
                preds = torch.argmax(logits, dim=-1)
                loss = self.criterion(logits, labels)

            # Store predictions and labels for later metrics
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Calculate batch accuracy
            batch_acc = (preds == labels).sum().item() / batch_size

            # Track batch loss and accuracy (per batch)
            total_loss += loss.item()
            total_acc += batch_acc

            # Update progress bar with current metrics
            batch_progress.set_postfix(
                {
                    "loss": loss.item(),
                    "acc": batch_acc,
                }
            )

        # Return average loss and accuracy per batch
        return total_loss / dataloader_len, total_acc / dataloader_len

    def detailed_evaluation(
        self, dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:
        """
        Perform a detailed evaluation on the given dataloader, including:
        - Loss and accuracy
        - Precision, recall, F1-score
        - Classification report
        - Confusion matrix

        Returns:
            Dict containing all evaluation metrics and results
        """
        self.model.eval()

        all_preds = []
        all_labels = []
        all_probs = []  # Store probabilities for ROC curve if needed later
        total_loss = 0.0

        batch_progress = tqdm(dataloader, desc="Detailed Evaluation")
        dataloader_len = len(dataloader)

        for batch in batch_progress:
            embeddings, labels = batch
            embeddings, labels = embeddings.to(self.device), labels.to(self.device)

            with torch.inference_mode():
                logits = self.model(embeddings)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                loss = self.criterion(logits, labels)

            # Store for metrics calculation
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            total_loss += loss.item()

        # Convert to numpy arrays for sklearn metrics
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)

        # Calculate metrics
        accuracy = (y_pred == y_true).mean()
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted"
        )

        # Generate classification report
        class_names = [self.class_mapping[i] for i in range(len(self.class_mapping))]
        report = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True
        )

        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Compile results
        results = {
            "loss": total_loss / dataloader_len,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "classification_report": report,
            "confusion_matrix": cm,
            "y_true": y_true,
            "y_pred": y_pred,
            "probabilities": np.array(all_probs),
        }

        return results

    def plot_confusion_matrix(self, cm: np.ndarray, save_path: str = None) -> None:
        """
        Plot confusion matrix.

        Args:
            cm: Confusion matrix from sklearn
            save_path: Path to save the plot, if None, just displays it
        """
        class_names = [self.class_mapping[i] for i in range(len(self.class_mapping))]

        # Create a larger figure for better visibility
        plt.figure(figsize=(10, 8))

        # Plot raw counts
        im = plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix", fontsize=16)
        plt.colorbar(im)

        # Remove grid
        plt.grid(False)

        # Add labels
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, fontsize=12)
        plt.yticks(tick_marks, class_names, fontsize=12)

        # Add text annotations for raw counts
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14,
                    fontweight="bold",
                )

        plt.ylabel("True label", fontsize=14)
        plt.xlabel("Predicted label", fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
