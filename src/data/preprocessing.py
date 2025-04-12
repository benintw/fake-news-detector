from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_df_from_config(config: dict[str, Any]) -> pd.DataFrame:
    """Load data from CSV files specified in the config.

    Args:
        config: Configuration dictionary

    Returns:
        DataFrame containing combined real and fake news
    """
    # Load data
    df_true = pd.read_csv(config["true_csv"])
    df_fake = pd.read_csv(config["false_csv"])

    # Add labels
    df_true["isfake"] = 0  # Real news
    df_fake["isfake"] = 1  # Fake news

    # Combine datasets
    df_combined = pd.concat([df_true, df_fake], axis=0).reset_index(drop=True)

    # Clean up and add features
    if "date" in df_combined.columns:
        df_combined.drop(columns=["date"], inplace=True)

    df_combined["isfake"] = df_combined["isfake"].astype(int)
    df_combined["original"] = df_combined["title"] + " " + df_combined["text"]

    return df_combined


def split_data(
    df: pd.DataFrame, config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, validation, and test sets.

    Args:
        df: Combined DataFrame to split
        config: Configuration dictionary

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Split into train+val and test sets
    train_val_df, test_df = train_test_split(
        df,
        test_size=config["train_test_split"],
        random_state=config["random_state"],
        stratify=df["isfake"],
    )

    # Split train+val into train and validation sets
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=config["train_val_split"],
        random_state=config["random_state"],
        stratify=train_val_df["isfake"],
    )

    print(f"Train set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")

    return train_df, val_df, test_df
