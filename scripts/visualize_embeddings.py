#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualize embeddings from the fake news detector model.

This script loads a dataset, computes embeddings, and visualizes them using
dimensionality reduction techniques (PCA and t-SNE).
"""

import os
import argparse
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wandb

from src.data.embeddings_processor import EmbeddingsProcessor
from src.visualization.embedding_visualizer import EmbeddingVisualizer
from src.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize embeddings from the fake news detector model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/dataset.yaml,configs/model.yaml,configs/training.yaml",
        help="Comma-separated list of config files"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="outputs/visualizations",
        help="Directory to save visualizations"
    )
    parser.add_argument(
        "--n_samples", 
        type=int, 
        default=500,
        help="Number of samples to visualize (use -1 for all)"
    )
    parser.add_argument(
        "--methods", 
        type=str, 
        default="pca,tsne",
        help="Comma-separated list of visualization methods"
    )
    parser.add_argument(
        "--use_wandb", 
        action="store_true",
        help="Log visualizations to Weights & Biases"
    )
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config_files = args.config.split(",")
    config = load_config(config_files)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.login()
        wandb.init(
            project=config.get("wandb_project", "fake-news-detector"),
            entity=config.get("wandb_entity", "chen-ben968"),
            name=config.get("wandb_run_name", "embedding-visualization"),
            config=config
        )
    
    # Load data
    print("Loading data...")
    true_df = pd.read_csv(config["true_csv"])
    fake_df = pd.read_csv(config["false_csv"])
    
    # Add labels
    true_df["isfake"] = 0
    fake_df["isfake"] = 1
    
    # Combine and sample data
    combined_df = pd.concat([true_df, fake_df], ignore_index=True)
    if args.n_samples > 0 and args.n_samples < len(combined_df):
        combined_df = combined_df.sample(args.n_samples, random_state=config.get("random_state", 42))
    
    # Initialize embeddings processor
    print("Computing embeddings...")
    embeddings_processor = EmbeddingsProcessor(config)
    
    # Compute embeddings
    embeddings = embeddings_processor.compute_embeddings(combined_df)
    
    # Convert embeddings to numpy array if they're torch tensors
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    
    # Get labels
    labels = combined_df["isfake"].values
    
    # Initialize visualizer
    print("Generating visualizations...")
    config["use_wandb"] = args.use_wandb  # Add wandb flag to config
    visualizer = EmbeddingVisualizer(config)
    
    # Parse visualization methods
    methods = args.methods.split(",")
    
    # Generate visualizations
    results = visualizer.visualize_embeddings(
        embeddings=embeddings,
        labels=labels,
        methods=methods,
        save_dir=args.output_dir
    )
    
    # Display results
    for method, result in results.items():
        plt.figure()
        result["plot"].show()
    
    print(f"Visualizations saved to {args.output_dir}")
    
    # Close wandb run if active
    if args.use_wandb and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
