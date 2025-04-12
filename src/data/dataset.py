from typing import List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


class NewsDataset(Dataset):
    """Dataset for fake news detection.

    This dataset handles both fake and real news articles with precomputed embeddings.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        embeddings: torch.Tensor,
        label_column: str = "isfake",
    ) -> None:
        """
        Initialize the news dataset with precomputed embeddings.

        Args:
            df: DataFrame containing news articles
            embeddings: Precomputed embeddings tensor
            label_column: Column name containing the label (1 for fake, 0 for real)
        """
        self.df = df
        self.embeddings = embeddings
        self.label_column = label_column

        # Verify that embeddings match dataframe length
        if len(df) != len(embeddings):
            raise ValueError(
                f"Embeddings length ({len(embeddings)}) doesn't match dataframe length ({len(df)})"
            )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset.

        Returns:
            Tuple containing:
                - document_embedding: Tensor representation of the text
                - label: Binary label (1 for fake news, 0 for real news)
        """
        # Get the label from the row at the specified index
        label = self.df.iloc[idx][self.label_column]

        # Return the precomputed embedding and the label
        return self.embeddings[idx], torch.tensor(label, dtype=torch.float16)
