from typing import Any

import torch
import torch.nn as nn


class FakeNewsClassifier(nn.Module):

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config

        self.classifier = nn.Sequential(
            nn.Linear(config["embed_dim"], config["hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["hidden_dim"], config["output_dim"]),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:

        logits = self.classifier(embedding)
        return logits
