from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification


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


class _HFTextDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, texts, y, max_length: int) -> None:
        self.enc = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.enc.items()}
        item["labels"] = self.labels[idx]
        return item


class BertTextClassifier:
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        batch_size: int = 8,
        epochs: int = 3,
        learning_rate: float = 2e-5,
        max_length: int = 256,
        weight_decay: float = 0.01,
    ):

        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.weight_decay = weight_decay

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def fit(self, train_texts, train_y, num_classes: int) -> None:
        

        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=num_classes).to(self.device)
        ds = _HFTextDataset(self.tokenizer, train_texts, train_y, self.max_length)
        loader = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        self.model.train()
        for _ in range(self.epochs):
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                out = self.model(**batch)
                loss = out.loss
                opt.zero_grad()
                loss.backward()
                opt.step()

    def predict(self, texts):
        if self.model is None:
            raise RuntimeError("BertTextClassifier must be fitted before predict.")
        
        ds = _HFTextDataset(self.tokenizer, texts, np.zeros(len(texts), dtype=np.int64), self.max_length)
        loader = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        
        self.model.eval()
        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
                logits = self.model(**batch).logits
                preds.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())
        return np.asarray(preds, dtype=np.int64)


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
