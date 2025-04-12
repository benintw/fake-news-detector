from typing import Any, Dict

import wandb
import yaml
from torch.utils.data import DataLoader

from .dataset import NewsDataset
from .embeddings_processor import get_df_with_embeddings, split_df_with_embeddings


def create_dataloader(
    dataset: NewsDataset, config: Dict[str, Any], is_train: bool = False
) -> DataLoader:
    """Create a DataLoader for a dataset.

    Args:
        dataset: The dataset to create a DataLoader for
        config: Configuration dictionary
        is_train: Whether this is a training dataset (affects shuffling)

    Returns:
        DataLoader for the dataset
    """
    return DataLoader(
        dataset,
        batch_size=config["BATCH_SIZE"],
        shuffle=is_train,  # Only shuffle training data
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )


def create_dataloaders(
    config: Dict[str, Any], force_recompute: bool = False
) -> Dict[str, DataLoader]:
    """Create DataLoader objects for train, validation, and test sets.

    This function handles the entire pipeline:
    1. Load/compute the dataframe with embeddings
    2. Split into train/val/test sets
    3. Create datasets with the appropriate embeddings
    4. Create and return dataloaders

    Args:
        config: Configuration dictionary
        force_recompute: Whether to force recomputation of embeddings

    Returns:
        Dictionary of DataLoader objects for train, val, and test sets
    """
    # Step 1: Get the dataframe with embeddings (computed once for all splits)
    df_combined, all_embeddings = get_df_with_embeddings(config, force_recompute)

    # Step 2: Split the dataframe and embeddings
    split_data = split_df_with_embeddings(df_combined, all_embeddings, config)

    # Step 3 & 4: Create datasets and dataloaders
    dataloaders = {}

    for split_name, (split_df, split_embeddings) in split_data.items():
        # Create dataset
        dataset = NewsDataset(
            df=split_df, embeddings=split_embeddings, label_column="isfake"
        )

        # Create dataloader
        is_train = split_name == "train"
        dataloaders[split_name] = create_dataloader(dataset, config, is_train)

    return dataloaders


def main() -> None:
    """Test the dataset implementation with the new embeddings-first approach."""
    # Initialize wandb
    try:
        wandb.login()
        wandb.init(project="fake-news-detector", entity="chen-ben968-benchen")
    except Exception as e:
        print(f"Warning: Could not initialize wandb: {e}")

    with open("./configs/dataset.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Create dataloaders with the new approach
    print("Creating dataloaders with embeddings-first approach...")
    dataloaders = create_dataloaders(config)

    print(len(dataloaders))
    print(len(dataloaders["train"]))
    print(len(dataloaders["val"]))
    print(len(dataloaders["test"]))

    # Test the dataloaders
    for split_name, dataloader in dataloaders.items():
        print(f"\n{split_name.capitalize()} dataloader:")
        batch = next(iter(dataloader))
        embeddings, labels = batch
        print(f"  Embeddings shape: {embeddings.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Label distribution: {labels.sum().item()}/{len(labels)} positive")

    print("\nDataloaders created successfully!")

    # Finish wandb run
    try:
        wandb.finish()
    except:
        pass


if __name__ == "__main__":
    main()
