import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Set style
sns.set_style("whitegrid")


class EmbeddingVisualizer:
    """
    Class for visualizing embeddings using dimensionality reduction techniques.
    """

    def __init__(self, config=None):
        """
        Initialize the visualizer.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.use_wandb = self.config.get("use_wandb", False)

        # Set default figure size
        plt.rcParams["figure.figsize"] = (12, 8)

    def visualize_with_pca(
        self, embeddings, labels=None, n_components=2, save_path=None
    ):
        """
        Visualize embeddings using PCA.

        Args:
            embeddings: Numpy array of embeddings
            labels: Optional labels for coloring points
            n_components: Number of PCA components
            save_path: Path to save the plot

        Returns:
            plt: Matplotlib figure
            reduced_embeddings: Reduced embeddings
        """
        # Apply PCA
        pca = PCA(n_components=n_components)
        reduced_embeddings = pca.fit_transform(embeddings)

        # Create dataframe for plotting
        plot_df = pd.DataFrame(
            reduced_embeddings, columns=[f"PC{i+1}" for i in range(n_components)]
        )

        # Add labels if available
        if labels is not None:
            # Get class mapping from config
            class_mapping = self.config.get("class_mapping", {0: "real", 1: "fake"})
            plot_df["Label"] = [
                class_mapping.get(int(label), str(label)) for label in labels
            ]

        # Create plot
        plt.figure(figsize=(10, 8))
        if labels is not None:
            sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="Label", palette="Set1")
        else:
            sns.scatterplot(data=plot_df, x="PC1", y="PC2")

        plt.title(
            f"PCA Visualization of Document Embeddings\nExplained Variance: {pca.explained_variance_ratio_.sum():.2f}"
        )
        plt.tight_layout()

        # Save plot if path provided
        if save_path:
            plt.savefig(save_path)

        # Log to wandb if enabled
        if self.use_wandb:
            try:
                wandb.log({"pca_plot": wandb.Image(plt)})
            except Exception as e:
                print(f"Warning: Could not log to wandb: {e}")

        return plt, reduced_embeddings

    def visualize_with_tsne(
        self, embeddings, labels=None, n_components=2, perplexity=30, save_path=None
    ):
        """
        Visualize embeddings using t-SNE.

        Args:
            embeddings: Numpy array of embeddings
            labels: Optional labels for coloring points
            n_components: Number of t-SNE components
            perplexity: t-SNE perplexity parameter
            save_path: Path to save the plot

        Returns:
            plt: Matplotlib figure
            reduced_embeddings: Reduced embeddings
        """
        # Apply t-SNE
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            max_iter=1000,
            random_state=42,
        )
        reduced_embeddings = tsne.fit_transform(embeddings)

        # Create dataframe for plotting
        plot_df = pd.DataFrame(
            reduced_embeddings, columns=[f"t-SNE{i+1}" for i in range(n_components)]
        )

        # Add labels if available
        if labels is not None:
            # Get class mapping from config
            class_mapping = self.config.get("class_mapping", {0: "real", 1: "fake"})
            plot_df["Label"] = [
                class_mapping.get(int(label), str(label)) for label in labels
            ]

        # Create plot
        plt.figure(figsize=(10, 8))
        if labels is not None:
            sns.scatterplot(
                data=plot_df, x="t-SNE1", y="t-SNE2", hue="Label", palette="Set1"
            )
        else:
            sns.scatterplot(data=plot_df, x="t-SNE1", y="t-SNE2")

        plt.title(
            f"t-SNE Visualization of Document Embeddings (perplexity={perplexity})"
        )
        plt.tight_layout()

        # Save plot if path provided
        if save_path:
            plt.savefig(save_path)

        # Log to wandb if enabled
        if self.use_wandb:
            try:
                wandb.log({"tsne_plot": wandb.Image(plt)})
            except Exception as e:
                print(f"Warning: Could not log to wandb: {e}")

        return plt, reduced_embeddings

    def visualize_embeddings(
        self, embeddings, labels=None, methods=None, save_dir=None
    ):
        """
        Visualize embeddings using multiple methods.

        Args:
            embeddings: Numpy array of embeddings
            labels: Optional labels for coloring points
            methods: List of methods to use ("pca", "tsne")
            save_dir: Directory to save plots

        Returns:
            Dictionary of plots and reduced embeddings
        """
        methods = methods or ["pca", "tsne"]
        results = {}

        for method in methods:
            if method == "pca":
                save_path = f"{save_dir}/pca_plot.png" if save_dir else None
                plt, reduced = self.visualize_with_pca(
                    embeddings, labels, save_path=save_path
                )
                results["pca"] = {"plot": plt, "embeddings": reduced}
            elif method == "tsne":
                save_path = f"{save_dir}/tsne_plot.png" if save_dir else None
                plt, reduced = self.visualize_with_tsne(
                    embeddings, labels, save_path=save_path
                )
                results["tsne"] = {"plot": plt, "embeddings": reduced}

        return results
