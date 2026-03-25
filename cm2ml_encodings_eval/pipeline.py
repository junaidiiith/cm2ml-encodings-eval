from __future__ import annotations

import os
from typing import Dict, List
from anyio import Path
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from .config import (
    BertClassifierConfig,
    GNNExperimentConfig,
    MetaDataConfig,
    SplitConfig,
    TextBuildConfig,
    TextExperimentConfig,
    TreeSerializationConfig,
)
from .encoders import get_encoder
from .graph_data import pyg_from_nx
from .metrics import classification_metrics
from .models import BertTextClassifier, ClassicTextClassifier, EdgeGNN, NodeGNN
from .split import train_test_indices
from .text import build_edge_corpus, build_node_corpus, node_text
from .tree_transform import serialize_graphs_to_trees


def _prepare_labels(y_raw: List[str]):
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    return y, le


def _as_graph_list(graphs_or_dataset) -> List:
    if graphs_or_dataset is None:
        return []
    if hasattr(graphs_or_dataset, "graphs"):
        return list(graphs_or_dataset.graphs)
    if isinstance(graphs_or_dataset, list):
        return graphs_or_dataset
    if isinstance(graphs_or_dataset, tuple):
        return list(graphs_or_dataset)
    try:
        return list(graphs_or_dataset)
    except TypeError:
        return [graphs_or_dataset]


def _to_dense_np(x) -> np.ndarray:
    if sp.issparse(x):
        return x.toarray().astype(np.float32)
    return np.asarray(x, dtype=np.float32)

def get_config_str(text_cfg: TextBuildConfig, exp_cfg: TextExperimentConfig, bert_cfg: BertClassifierConfig | None = None) -> str:
    cfg_parts = [
        f"metadata_cls={text_cfg.metadata.cls}",
        f"include_node_label={text_cfg.include_node_label}",
        f"include_node_type={text_cfg.include_node_type}",
        f"include_node_attributes={text_cfg.include_node_attributes}",
        f"include_edge_label={text_cfg.include_edge_label}",
        f"include_edge_type={text_cfg.include_edge_type}",
        f"path_depth={text_cfg.path_depth}",
        f"max_paths_per_node={text_cfg.max_paths_per_node}",
        f"classifier={exp_cfg.classifier_name}",
        f"encoder={exp_cfg.encoder_name}",
    ]
    if exp_cfg.classifier_name.lower() == "bert_classifier" and bert_cfg is not None:
        cfg_parts.extend([
            f"bert_model={bert_cfg.model_name}",
            f"bert_batch_size={bert_cfg.batch_size}",
            f"bert_epochs={bert_cfg.epochs}",
            f"bert_learning_rate={bert_cfg.learning_rate}",
            f"bert_max_length={bert_cfg.max_length}",
            f"bert_weight_decay={bert_cfg.weight_decay}",
        ])
    config_str = "__".join(cfg_parts)
    if text_cfg.is_tree:
        config_str += "__tree"
    return config_str

def get_corpus_path(text_cfg: TextBuildConfig, exp_cfg: TextExperimentConfig, bert_cfg: BertClassifierConfig | None = None) -> Path:
    config_str = get_config_str(text_cfg, exp_cfg, bert_cfg)
    config_str_hash = str(abs(hash(config_str)) % (10 ** 8))  # Short hash for readability
    corpus_path = Path(f"corpus_pickles/{config_str_hash}.npz")
    return corpus_path

def node_corpus_exists(text_cfg: TextBuildConfig, exp_cfg: TextExperimentConfig, bert_cfg: BertClassifierConfig | None = None) -> bool:
    corpus_path = get_corpus_path(text_cfg, exp_cfg, bert_cfg)
    return os.path.exists(corpus_path)

def load_node_corpus(text_cfg: TextBuildConfig, exp_cfg: TextExperimentConfig, bert_cfg: BertClassifierConfig | None = None):
    corpus_path = get_corpus_path(text_cfg, exp_cfg, bert_cfg)
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"No existing corpus found at {corpus_path}")
    
    data = np.load(corpus_path, allow_pickle=True)
    print(f"Loaded saved node corpus from {corpus_path}...")
    
    texts = data['texts'].item()
    y_raw = data['y_raw'].item()
    indices = data['indices'].item()
    return texts, y_raw, indices

def save_node_corpus(text_cfg: TextBuildConfig, exp_cfg: TextExperimentConfig, bert_cfg: BertClassifierConfig | None, texts, y_raw, indices):
    print("Saving node corpus for future runs...")
    corpus_path = get_corpus_path(text_cfg, exp_cfg, bert_cfg)
    np.savez(corpus_path, texts=texts, y_raw=y_raw, indices=indices)

def run_text_node_classification(
    graphs,
    text_cfg: TextBuildConfig | None = None,
    exp_cfg: TextExperimentConfig | None = None,
    bert_cfg: BertClassifierConfig | None = None,
) -> Dict:
    text_cfg = text_cfg or TextBuildConfig()
    exp_cfg = exp_cfg or TextExperimentConfig()
    
    if node_corpus_exists(text_cfg, exp_cfg, bert_cfg):
        print("Loading existing node text corpus...")
        texts, y_raw, indices = load_node_corpus(text_cfg, exp_cfg, bert_cfg)
    else:
        print("Building node text corpus...")
        texts, y_raw, indices = build_node_corpus(
            graphs,
            metadata=text_cfg.metadata,
            include_node_label=text_cfg.include_node_label,
            include_node_type=text_cfg.include_node_type,
            include_node_attributes=text_cfg.include_node_attributes,
            include_edge_label=text_cfg.include_edge_label,
            include_edge_type=text_cfg.include_edge_type,
            path_depth=text_cfg.path_depth,
            max_paths_per_node=text_cfg.max_paths_per_node,
        )
        
        save_node_corpus(text_cfg, exp_cfg, bert_cfg, texts, y_raw, indices)
        
    if not texts:
        raise ValueError("No node samples found. Check target_attr and graph content.")

    _, label_enc = _prepare_labels([lbl for lbl in y_raw['train'] + y_raw['test']])
    y_train = label_enc.transform(y_raw['train'])
    y_test = label_enc.transform(y_raw['test'])
    # train_idx, test_idx = train_test_indices(y, split_cfg.test_ratio, split_cfg.random_state, split_cfg.stratify)

    if exp_cfg.classifier_name.lower() == "bert_classifier":
        cfg = bert_cfg or BertClassifierConfig()
        clf = BertTextClassifier(
            model_name=cfg.model_name,
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            learning_rate=cfg.learning_rate,
            max_length=cfg.max_length,
            weight_decay=cfg.weight_decay,
        )
        print("Training BERT classifier...")
        clf.fit(texts['train'], y_train, num_classes=len(label_enc.classes_))
        print("Predicting test set with BERT classifier...")
        pred = clf.predict(texts['test'])
    else:
        encoder = get_encoder(exp_cfg.encoder_name)
        print(f"Encoding text with {exp_cfg.encoder_name}...")
        x_train = encoder.fit_transform(texts['train'])
        x_test = encoder.transform(texts['test'])
        clf = ClassicTextClassifier(exp_cfg.classifier_name)
        print(f"Training classic text classifier... ({exp_cfg.classifier_name}) with {x_train.shape[0]} samples")
        clf.fit(x_train, y_train)
        pred = clf.predict(x_test)

    m = classification_metrics(y_test, pred)
    return {
        "task": "node_text_classification",
        "target_attr": text_cfg.metadata.cls,
        "num_samples": len(texts['train']) + len(texts['test']),
        "num_classes": len(label_enc.classes_),
        "metrics": m,
        # "test_indices": [indices[i] for i in range(len(indices)) if i in [j[0] for j in indices if j[1] in texts['test']]],
        # "y_true": y_test.tolist(),
        # "y_pred": pred.tolist(),
        "class_names": label_enc.classes_.tolist(),
    }


def run_text_edge_classification(
    graphs,
    target_attr: str,
    split_cfg: SplitConfig | None = None,
    text_cfg: TextBuildConfig | None = None,
    exp_cfg: TextExperimentConfig | None = None,
    bert_cfg: BertClassifierConfig | None = None,
) -> Dict:
    split_cfg = split_cfg or SplitConfig()
    text_cfg = text_cfg or TextBuildConfig()
    exp_cfg = exp_cfg or TextExperimentConfig()
    texts, y_raw, indices = build_edge_corpus(
        graphs,
        metadata=text_cfg.metadata,
        include_node_label=text_cfg.include_node_label,
        include_node_type=text_cfg.include_node_type,
        include_node_attributes=text_cfg.include_node_attributes,
        include_edge_label=text_cfg.include_edge_label,
        include_edge_type=text_cfg.include_edge_type,
        path_depth=text_cfg.path_depth,
        max_paths_per_node=text_cfg.max_paths_per_node,
    )
    if not texts:
        raise ValueError("No edge samples found. Check target_attr and graph content.")

    y, label_enc = _prepare_labels(y_raw)
    train_idx, test_idx = train_test_indices(y, split_cfg.test_ratio, split_cfg.random_state, split_cfg.stratify)

    if exp_cfg.classifier_name.lower() == "bert_classifier":
        cfg = bert_cfg or BertClassifierConfig()
        clf = BertTextClassifier(
            model_name=cfg.model_name,
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            learning_rate=cfg.learning_rate,
            max_length=cfg.max_length,
            weight_decay=cfg.weight_decay,
        )
        clf.fit([texts[i] for i in train_idx], y[train_idx], num_classes=len(label_enc.classes_))
        pred = clf.predict([texts[i] for i in test_idx])
    else:
        encoder = get_encoder(exp_cfg.encoder_name)
        x_train = encoder.fit_transform([texts[i] for i in train_idx])
        x_test = encoder.transform([texts[i] for i in test_idx])
        clf = ClassicTextClassifier(exp_cfg.classifier_name)
        clf.fit(x_train, y[train_idx])
        pred = clf.predict(x_test)

    m = classification_metrics(y[test_idx], pred)
    return {
        "task": "edge_text_classification",
        "target_attr": target_attr,
        "num_samples": len(texts),
        "num_classes": len(label_enc.classes_),
        "metrics": m,
        # "test_indices": [indices[i] for i in test_idx],
        # "y_true": y[test_idx].tolist(),
        # "y_pred": pred.tolist(),
        "class_names": label_enc.classes_.tolist(),
    }


def run_gnn_node_classification(
    graphs_or_dataset,
    node_encoder_name: str = "tfidf",
    text_cfg: TextBuildConfig | None = None,
    gnn_cfg: GNNExperimentConfig | None = None,
) -> Dict:
    text_cfg = text_cfg or TextBuildConfig()
    gnn_cfg = gnn_cfg or GNNExperimentConfig()
    graphs = _as_graph_list(graphs_or_dataset)
    if not graphs:
        raise ValueError("No graphs provided. Pass a ModelDataset or an iterable of NetworkX graphs.")

    # Fit one global label space across all graphs so class ids are consistent in batched training.
    all_labels: List[str] = []
    for graph in graphs:
        for _, data in graph.nodes(data=True):
            lbl = data.get(text_cfg.metadata.cls)
            if lbl is not None:
                all_labels.append(str(lbl))
    if not all_labels:
        raise ValueError(f"No labeled nodes found for metadata class '{text_cfg.metadata.cls}' in dataset.")

    global_le = LabelEncoder()
    global_le.fit(all_labels)
    label_to_global = {label: idx for idx, label in enumerate(global_le.classes_)}

    global_encoder = get_encoder(node_encoder_name)
    all_node_texts = dict()
    for i, graph in enumerate(graphs):
        graph_texts = [
            node_text(
                node_data,
                text_cfg.metadata,
                text_cfg.include_node_label,
                text_cfg.include_node_type,
                text_cfg.include_node_attributes,
            ) for _, node_data in graph.nodes(data=True)    
        ]
        all_node_texts[i] = graph_texts
        
        
    global_encoder.fit([t for texts in all_node_texts.values() for t in texts])

    pyg_graphs = []
    for i, graph in enumerate(graphs):
        node_ids = list(graph.nodes())
        if not node_ids:
            continue
        node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}

        graph_texts = all_node_texts[i]
        x_np = _to_dense_np(global_encoder.transform(graph_texts))
        x = torch.tensor(x_np, dtype=torch.float32)

        edges = [(node_to_idx[u], node_to_idx[v]) for u, v in graph.edges()]
        edge_index = (
            torch.tensor(edges, dtype=torch.long).t().contiguous()
            if edges
            else torch.empty((2, 0), dtype=torch.long)
        )

        y = torch.full((len(node_ids),), -1, dtype=torch.long)
        for i, n in enumerate(node_ids):
            raw_label = graph.nodes[n].get(text_cfg.metadata.cls)
            if raw_label is None:
                continue
            y[i] = label_to_global[str(raw_label)]

        test_mask = torch.tensor(
            [bool(graph.nodes[n].get("masked", False)) for n in node_ids],
            dtype=torch.bool,
        )
        train_mask = torch.tensor(
            [True if 'masked' in graph.nodes[n] and graph.nodes[n]['masked'] == False else False for n in node_ids],
            dtype=torch.bool,
        )
        # train_mask = ~test_mask

        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            test_mask=test_mask,
        )

        # Keep only graphs that contribute both train and test labeled nodes.
        train_valid = bool(((data.train_mask) & (data.y >= 0)).any().item())
        test_valid = bool(((data.test_mask) & (data.y >= 0)).any().item())
        if train_valid and test_valid:
            pyg_graphs.append(data)

    if not pyg_graphs:
        raise ValueError("No usable graphs after preprocessing. Ensure each graph has train/test masked labeled nodes.")

    in_dim = int(pyg_graphs[0].x.size(1))
    out_dim = len(global_le.classes_)
    batch_size = 1 if in_dim > 1000 else 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(pyg_graphs, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(pyg_graphs, batch_size=batch_size, shuffle=False)

    

    model = NodeGNN(
        in_dim=in_dim,
        hidden_dim=gnn_cfg.hidden_dim,
        out_dim=out_dim,
        model_name=gnn_cfg.model_name,
        num_layers=gnn_cfg.num_layers,
        dropout=gnn_cfg.dropout,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=gnn_cfg.learning_rate, weight_decay=gnn_cfg.weight_decay)

    for _ in tqdm(range(gnn_cfg.epochs), desc="Training GNN"):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            train_mask = batch.train_mask & (batch.y >= 0)
            if not bool(train_mask.any().item()):
                continue
            loss = torch.nn.functional.cross_entropy(out[train_mask], batch.y[train_mask])
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    y_true_all = []
    y_pred_all = []
    with torch.no_grad():
        for batch in eval_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index)
            pred = logits.argmax(dim=1)
            test_mask = batch.test_mask & (batch.y >= 0)
            if not bool(test_mask.any().item()):
                continue
            y_true_all.append(batch.y[test_mask].detach().cpu())
            y_pred_all.append(pred[test_mask].detach().cpu())

    if not y_true_all:
        raise ValueError("Evaluation failed: no test nodes found across dataset. Ensure masked test nodes are present.")

    y_true = torch.cat(y_true_all).numpy()
    y_pred = torch.cat(y_pred_all).numpy()
    m = classification_metrics(y_true, y_pred)

    return {
        "task": "gnn_node_classification",
        "target_attr": text_cfg.metadata.cls,
        "num_graphs": len(pyg_graphs),
        "num_nodes": int(sum(g.x.size(0) for g in pyg_graphs)),
        "num_classes": len(global_le.classes_),
        "metrics": m,
        "class_names": global_le.classes_.tolist(),
    }


def run_gnn_edge_classification(
    graph,
    metadata: MetaDataConfig,
    node_encoder_name: str = "tfidf",
    edge_encoder_name: str | None = "tfidf",
    split_cfg: SplitConfig | None = None,
    text_cfg: TextBuildConfig | None = None,
    gnn_cfg: GNNExperimentConfig | None = None,
) -> Dict:
    split_cfg = split_cfg or SplitConfig()
    text_cfg = text_cfg or TextBuildConfig()
    gnn_cfg = gnn_cfg or GNNExperimentConfig()
    node_encoder = get_encoder(node_encoder_name)
    edge_encoder = get_encoder(edge_encoder_name) if edge_encoder_name else None
    data, le = pyg_from_nx(
        graph,
        node_encoder=node_encoder,
        edge_encoder=edge_encoder,
        metadata=metadata.cls,
        task="edge",
        include_node_label=text_cfg.include_node_label,
        include_node_type=text_cfg.include_node_type,
        include_node_attributes=text_cfg.include_node_attributes,
        include_edge_label=text_cfg.include_edge_label,
        include_edge_type=text_cfg.include_edge_type,
        split=(split_cfg.test_ratio, split_cfg.random_state),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    edge_dim = 0 if data.edge_attr is None else int(data.edge_attr.size(1))
    model = EdgeGNN(
        in_dim=int(data.x.size(1)),
        hidden_dim=gnn_cfg.hidden_dim,
        out_dim=len(le.classes_),
        edge_dim=edge_dim,
        model_name=gnn_cfg.model_name,
        num_layers=gnn_cfg.num_layers,
        dropout=gnn_cfg.dropout,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=gnn_cfg.learning_rate, weight_decay=gnn_cfg.weight_decay)

    for _ in range(gnn_cfg.epochs):
        model.train()
        logits = model(data.x, data.edge_index, data.edge_pairs, data.edge_attr)
        loss = torch.nn.functional.cross_entropy(logits[data.edge_train_mask], data.edge_y[data.edge_train_mask])
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_pairs, data.edge_attr)
        pred = logits.argmax(dim=1)

    y_true = data.edge_y[data.edge_test_mask].detach().cpu().numpy()
    y_pred = pred[data.edge_test_mask].detach().cpu().numpy()
    m = classification_metrics(y_true, y_pred)

    return {
        "task": "gnn_edge_classification",
        "target_attr": metadata.cls,
        "num_edges_labeled": int(data.edge_y.size(0)),
        "num_classes": len(le.classes_),
        "metrics": m,
        "class_names": le.classes_.tolist(),
    }


def run_tree_text_node_classification(
    graphs,
    text_cfg: TextBuildConfig | None = None,
    exp_cfg: TextExperimentConfig | None = None,
    bert_cfg: BertClassifierConfig | None = None,
    tree_cfg: TreeSerializationConfig | None = None,
) -> Dict:
    text_cfg = text_cfg or TextBuildConfig()
    print("Serializing graphs to tree structures...")
    tree_graphs = serialize_graphs_to_trees(
        graphs,
        node_target_attr=text_cfg.metadata.cls,
        tree_cfg=tree_cfg,
    )
    print(f"Serialized {len(tree_graphs)} graphs into tree structures. Running text node classification on trees...")
    result = run_text_node_classification(
        tree_graphs,
        text_cfg=text_cfg,
        exp_cfg=exp_cfg,
        bert_cfg=bert_cfg,
    )
    result["task"] = "tree_node_text_classification"
    result["graph_transform"] = "model_obj_atts_assoc_tree"
    return result


def run_tree_gnn_node_classification(
    graphs_or_dataset,
    node_encoder_name: str = "tfidf",
    text_cfg: TextBuildConfig | None = None,
    gnn_cfg: GNNExperimentConfig | None = None,
    tree_cfg: TreeSerializationConfig | None = None,
) -> Dict:
    text_cfg = text_cfg or TextBuildConfig()
    graphs = _as_graph_list(graphs_or_dataset)
    if not graphs:
        raise ValueError("No graphs provided for tree GNN node classification.")

    tree_graphs = serialize_graphs_to_trees(
        graphs,
        node_target_attr=text_cfg.metadata.cls,
        tree_cfg=tree_cfg,
    )
    result = run_gnn_node_classification(
        tree_graphs,
        node_encoder_name=node_encoder_name,
        text_cfg=text_cfg,
        gnn_cfg=gnn_cfg,
    )
    result["task"] = "tree_gnn_node_classification"
    result["graph_transform"] = "model_obj_atts_assoc_tree"
    return result
