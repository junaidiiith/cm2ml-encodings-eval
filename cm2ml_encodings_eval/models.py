from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC



class ClassicTextClassifier:
    def __init__(self, model_name: str = "logreg") -> None:
        m = model_name.lower()
        if m in {"logreg", "logistic", "lr"}:
            self.model = LogisticRegression(max_iter=3000)
        elif m in {"linearsvm", "svm", "linear_svm"}:
            self.model = LinearSVC()
        else:
            raise ValueError(f"Unsupported classifier: {model_name}")

    def fit(self, x_train, y_train) -> None:
        self.model.fit(x_train, y_train)

    def predict(self, x):
        return self.model.predict(x)


class NodeGNN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, model_name: str = "gcn", num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        from torch_geometric.nn import GATConv, GCNConv, SAGEConv

        model_name = model_name.lower()
        conv_map = {
            "gcn": GCNConv,
            "graphsage": SAGEConv,
            "sage": SAGEConv,
            "gat": GATConv,
        }
        if model_name not in conv_map:
            raise ValueError(f"Unsupported GNN model: {model_name}")
        conv_cls = conv_map[model_name]

        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(conv_cls(in_dim, out_dim))
        else:
            self.layers.append(conv_cls(in_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.layers.append(conv_cls(hidden_dim, hidden_dim))
            self.layers.append(conv_cls(hidden_dim, out_dim))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class EdgeGNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        edge_dim: int = 0,
        model_name: str = "gcn",
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = NodeGNN(in_dim, hidden_dim, hidden_dim, model_name=model_name, num_layers=num_layers, dropout=dropout)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.edge_dim = edge_dim

    def forward(self, x, edge_index, edge_pairs, edge_attr: Optional[torch.Tensor] = None):
        h = self.encoder(x, edge_index)
        src = h[edge_pairs[:, 0]]
        dst = h[edge_pairs[:, 1]]
        if self.edge_dim > 0 and edge_attr is not None:
            z = torch.cat([src, dst, edge_attr], dim=1)
        else:
            z = torch.cat([src, dst], dim=1)
        return self.edge_mlp(z)
