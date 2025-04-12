import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..data.embeddings_processor import EmbeddingsProcessor
from ..model.model import FakeNewsClassifier


class NewsPredictor:
    def __init__(self, config, model_path=None):
        self.config = config

        # Set device - try to use MPS (Apple Silicon) or CUDA if available, otherwise CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print(f"Using device: {self.device}")

        # Initialize model
        self.model = FakeNewsClassifier(config)
        self.model.to(self.device)

        # Load model weights
        if model_path is None:
            # Use default path
            checkpoint_dir = Path(config.get("output_dir", "./outputs")) / "checkpoints"
            model_path = checkpoint_dir / "best_accuracy_model.pth"

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded model from {model_path}")
        else:
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

        # Set model to evaluation mode
        self.model.eval()

        # Initialize embeddings processor
        self.embeddings_processor = EmbeddingsProcessor(config)

        # Get class mapping
        self.class_mapping = config.get("class_mapping", {0: "real", 1: "fake"})

    def _get_certainty_level(self, prob_diff):
        """
        Convert probability difference to a human-readable certainty level.

        Args:
            prob_diff: Absolute difference between real and fake probabilities

        Returns:
            String describing the certainty level
        """
        if prob_diff > 0.8:
            return "Very High"
        elif prob_diff > 0.6:
            return "High"
        elif prob_diff > 0.4:
            return "Moderate"
        elif prob_diff > 0.2:
            return "Low"
        else:
            return "Very Low"

    def predict_text(self, text):
        """
        Predict whether a single news article is real or fake.

        Args:
            text: The text of the news article

        Returns:
            Dictionary with prediction results
        """
        if not text or not text.strip():
            raise ValueError(
                "Empty text provided. Please provide a non-empty news article."
            )

        # Create a temporary dataframe with the text
        df = pd.DataFrame({"text": [text]})

        # Compute embeddings
        embeddings = self.embeddings_processor._compute_embeddings(df)

        # Move to device
        embeddings = embeddings.to(self.device)

        # Make prediction
        with torch.inference_mode():
            logits = self.model(embeddings)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1).item()

        # Get class probabilities
        class_probs = {
            self.class_mapping[i]: float(probs[0, i].item())
            for i in range(len(self.class_mapping))
        }

        # Calculate certainty metrics
        prob_diff = abs(class_probs["real"] - class_probs["fake"])
        certainty_level = self._get_certainty_level(prob_diff)

        # Return prediction results
        return {
            "prediction": self.class_mapping[pred],
            "probabilities": class_probs,
            "confidence": float(probs[0, pred].item()),
            "certainty": certainty_level,
            "text_length": len(text),
            "text_preview": text[:100] + "..." if len(text) > 100 else text,
        }

    def predict_file(self, file_path):
        """
        Predict whether news articles in a file are real or fake.

        Args:
            file_path: Path to a CSV or TXT file with news articles

        Returns:
            List of dictionaries with prediction results
        """
        # Load file
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
            if "text" not in df.columns:
                raise ValueError("CSV file must have a 'text' column")
            texts = df["text"].tolist()
        elif file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                # Check for a special delimiter that indicates multiple articles
                # Using a very specific delimiter to avoid accidental splits
                if "===ARTICLE_SEPARATOR===" in content:
                    texts = [
                        text.strip()
                        for text in content.split("===ARTICLE_SEPARATOR===")
                        if text.strip()
                    ]
                else:
                    # Treat the entire file as a single article by default
                    texts = [content]
        else:
            raise ValueError("File must be a CSV or TXT file")

        # Make predictions for each text
        results = []
        if len(texts) > 1:
            # Show progress bar for multiple articles
            for text in tqdm(texts, desc="Analyzing articles"):
                if text.strip():  # Skip empty texts
                    results.append(self.predict_text(text))
        else:
            # Single article, no need for progress bar
            for text in texts:
                if text.strip():  # Skip empty texts
                    results.append(self.predict_text(text))

        # Add metadata to results
        for result in results:
            result["timestamp"] = time.time()
            result["file_source"] = file_path

        return results

    def batch_predict(self, texts):
        """
        Predict whether multiple news articles are real or fake in a single batch.

        Args:
            texts: List of text strings to classify

        Returns:
            List of dictionaries with prediction results
        """
        if not texts:
            return []

        # Filter out empty texts
        valid_texts = [text for text in texts if text and text.strip()]
        if not valid_texts:
            return []

        # Create dataframe with texts
        df = pd.DataFrame({"text": valid_texts})

        # Process all texts in a single batch for efficiency
        all_embeddings = []

        # Process in smaller batches to avoid memory issues
        batch_size = min(32, len(valid_texts))
        for i in range(0, len(valid_texts), batch_size):
            batch_df = df.iloc[i : i + batch_size]
            batch_embeddings = self.embeddings_processor._compute_embeddings(batch_df)
            all_embeddings.append(batch_embeddings)

        # Combine all embeddings
        embeddings = torch.cat(all_embeddings, dim=0).to(self.device)

        # Make predictions for all texts at once
        with torch.inference_mode():
            logits = self.model(embeddings)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

        # Prepare results
        results = []
        for i, (text, pred, prob) in enumerate(zip(valid_texts, preds, probs)):
            pred_idx = pred.item()

            # Get class probabilities
            class_probs = {
                self.class_mapping[j]: float(prob[j].item())
                for j in range(len(self.class_mapping))
            }

            # Calculate certainty metrics
            prob_diff = abs(class_probs["real"] - class_probs["fake"])
            certainty_level = self._get_certainty_level(prob_diff)

            results.append(
                {
                    "prediction": self.class_mapping[pred_idx],
                    "probabilities": class_probs,
                    "confidence": float(prob[pred_idx].item()),
                    "certainty": certainty_level,
                    "text_length": len(text),
                    "text_preview": text[:100] + "..." if len(text) > 100 else text,
                }
            )

        return results
