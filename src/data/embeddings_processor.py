import hashlib
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Set tokenizers parallelism to false to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
from icecream import ic
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from .tokenization import sliding_window_tokenize


class EmbeddingsProcessor:
    """Process and manage embeddings for the fake news dataset.

    This class handles the computation, storage, and retrieval of embeddings
    for text data, allowing for efficient reuse across multiple experiments.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        cache_dir: str = "./cache",
        force_recompute: bool = False,
    ):
        """Initialize the embeddings processor.

        Args:
            config: Configuration dictionary
            cache_dir: Directory to cache embeddings and processed dataframes
            force_recompute: If True, recompute embeddings even if cached
        """
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.force_recompute = force_recompute
        # Force CPU usage to avoid memory issues
        self.device = torch.device("cpu")
        print(f"Using device: {self.device} (forced CPU mode for stability)")

        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

        # Load model and tokenizer
        self.model_name = config["model_name"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)

        # Set up paths
        self.df_cache_key = self._generate_df_cache_key()
        self.df_cache_path = self.cache_dir / f"{self.df_cache_key}.pkl"
        self.embeddings_cache_path = (
            self.cache_dir / f"{self.df_cache_key}_embeddings.pt"
        )

    def _generate_df_cache_key(self) -> str:
        """Generate a unique cache key for the dataframe and embeddings."""
        # Create a string representation of key configuration properties
        key_parts = [
            f"model_{self.model_name}",
            f"max_length_{self.config['max_length']}",
            f"stride_{self.config.get('stride', 128)}",
        ]

        # Add data sources to the key
        for source in ["true_csv", "false_csv"]:
            if source in self.config:
                file_path = self.config[source]
                # Get file modification time to detect changes
                if os.path.exists(file_path):
                    mtime = os.path.getmtime(file_path)
                    key_parts.append(f"{source}_{mtime}")

        # Combine all parts and hash
        key_str = "_".join(key_parts)
        return f"df_with_embeddings_{hashlib.md5(key_str.encode()).hexdigest()[:10]}"

    def get_processed_dataframe(self) -> pd.DataFrame:
        """Get the processed dataframe with embeddings.

        This will either load the cached dataframe or create a new one
        with computed embeddings.

        Returns:
            DataFrame with embeddings as a column
        """
        # Check if processed dataframe is cached
        if not self.force_recompute and self.df_cache_path.exists():
            print(f"Loading cached dataframe from {self.df_cache_path}...")
            try:
                with open(self.df_cache_path, "rb") as f:
                    df = pickle.load(f)
                print(f"Loaded dataframe with {len(df)} rows")
                return df
            except Exception as e:
                print(f"Error loading cached dataframe: {e}")

        # Process the dataframe if not cached or cache is invalid
        from .preprocessing import prepare_df_from_config

        df = prepare_df_from_config(self.config)
        ic(df.shape)
        # Compute and add embeddings
        print("Computing embeddings for all texts (this may take a while)...")
        embeddings_tensor = self._compute_embeddings(df)

        # Save the embeddings tensor separately (more efficient for loading)
        torch.save(embeddings_tensor, self.embeddings_cache_path)

        # Add embedding indices to the dataframe
        # We don't store the actual embeddings in the dataframe as they're large
        # Instead, we store indices that can be used to look up embeddings
        df["embedding_idx"] = range(len(df))

        # Save the processed dataframe
        print(f"Saving processed dataframe to {self.df_cache_path}...")
        with open(self.df_cache_path, "wb") as f:
            pickle.dump(df, f)

        return df

    def get_embeddings(self) -> torch.Tensor:
        """Get the embeddings tensor for all texts.

        Returns:
            Tensor of shape [num_samples, embedding_dim]
        """
        if self.embeddings_cache_path.exists():
            return torch.load(self.embeddings_cache_path)
        else:
            # If embeddings don't exist separately, reprocess the dataframe
            # which will create the embeddings file
            self.get_processed_dataframe()
            return torch.load(self.embeddings_cache_path)

    def _compute_embeddings(self, df: pd.DataFrame) -> torch.Tensor:
        """Compute embeddings for all texts in the dataframe.

        Args:
            df: DataFrame containing the texts

        Returns:
            Tensor of shape [num_samples, embedding_dim]
        """
        batch_size = self.config["BATCH_SIZE"]
        all_embeddings = []
        
        # Handle both column naming conventions (for training vs. inference)
        text_column = "original" if "original" in df.columns else "text"
        texts = df[text_column].tolist()

        # Create a progress bar
        total_batches = (len(texts) + batch_size - 1) // batch_size
        pbar = tqdm(total=total_batches, desc="Computing embeddings")

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_embeddings = []

            for text in batch_texts:
                # Get tokenized input with sliding window
                tokenized_input = sliding_window_tokenize(
                    text,
                    self.tokenizer,
                    max_length=self.config["max_length"],
                    stride=self.config.get("stride", 128),
                )

                # Process each window
                window_embeddings = []
                for j in range(len(tokenized_input["input_ids"])):
                    # Extract inputs for model
                    window_inputs = {
                        "input_ids": tokenized_input["input_ids"][j]
                        .unsqueeze(0)
                        .to(self.device),
                        "attention_mask": tokenized_input["attention_mask"][j]
                        .unsqueeze(0)
                        .to(self.device),
                    }

                    # Get model output
                    with torch.no_grad():
                        output = self.model(**window_inputs)

                    # Get embedding from [CLS] token
                    window_embedding = output.last_hidden_state[:, 0, :]
                    window_embeddings.append(window_embedding)

                # Average all window embeddings
                if window_embeddings:
                    doc_embedding = torch.mean(torch.cat(window_embeddings), dim=0)
                else:
                    # Fallback if no windows were created
                    doc_embedding = torch.zeros(self.model.config.hidden_size)

                batch_embeddings.append(doc_embedding)

            all_embeddings.extend(batch_embeddings)
            pbar.update(1)

        pbar.close()

        ic(torch.stack(all_embeddings).shape)
        # Stack all embeddings into a single tensor
        return torch.stack(all_embeddings).to(self.device)


def get_df_with_embeddings(
    config: Dict[str, Any], force_recompute: bool = False
) -> Tuple[pd.DataFrame, torch.Tensor]:
    """Get a dataframe with embedding indices and the corresponding embeddings tensor.

    This is a convenience function that handles the entire process of:
    1. Loading/preparing the dataframe
    2. Computing embeddings if needed
    3. Caching results for future use

    Args:
        config: Configuration dictionary
        force_recompute: If True, recompute embeddings even if cached

    Returns:
        Tuple containing:
            - DataFrame with an embedding_idx column
            - Tensor of embeddings where df['embedding_idx'] can be used as indices
    """
    processor = EmbeddingsProcessor(config, force_recompute=force_recompute)
    df = processor.get_processed_dataframe()
    embeddings = processor.get_embeddings()

    return df, embeddings


def split_df_with_embeddings(
    df: pd.DataFrame, embeddings: torch.Tensor, config: Dict[str, Any]
) -> Dict[str, Tuple[pd.DataFrame, torch.Tensor]]:
    """Split a dataframe with embeddings into train, validation, and test sets.

    Args:
        df: DataFrame with embedding_idx column
        embeddings: Tensor of embeddings
        config: Configuration dictionary with split ratios

    Returns:
        Dictionary with 'train', 'val', and 'test' keys, each containing a tuple of
        (dataframe, embeddings_tensor) for that split
    """
    from .preprocessing import split_data

    # Split the dataframe
    train_df, val_df, test_df = split_data(df, config)
    ic(train_df.shape, val_df.shape, test_df.shape)
    # Get the embeddings for each split using the embedding_idx column
    train_embeddings = embeddings[train_df["embedding_idx"].values]
    val_embeddings = embeddings[val_df["embedding_idx"].values]
    test_embeddings = embeddings[test_df["embedding_idx"].values]

    return {
        "train": (train_df, train_embeddings),
        "val": (val_df, val_embeddings),
        "test": (test_df, test_embeddings),
    }
