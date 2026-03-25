from __future__ import annotations

from pathlib import Path
from typing import List, Optional
from tqdm.auto import tqdm

import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder


class BaseTextEncoder:
    def fit(self, texts: List[str]) -> "BaseTextEncoder":
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)


class OneHotTextEncoder(BaseTextEncoder):
    def __init__(self) -> None:
        self.enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    def fit(self, texts: List[str]) -> "OneHotTextEncoder":
        self.enc.fit(np.array(texts, dtype=object).reshape(-1, 1))
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        return self.enc.transform(np.array(texts, dtype=object).reshape(-1, 1)).astype(np.float32)


class BoWTextEncoder(BaseTextEncoder):
    def __init__(self, max_features: Optional[int] = 50000) -> None:
        self.vec = CountVectorizer(max_features=max_features)

    def fit(self, texts: List[str]) -> "BoWTextEncoder":
        self.vec.fit(texts)
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        return self.vec.transform(texts).astype(np.float32)


class TfidfTextEncoder(BaseTextEncoder):
    def __init__(self, max_features: Optional[int] = 50000) -> None:
        self.vec = TfidfVectorizer(max_features=max_features)

    def fit(self, texts: List[str]) -> "TfidfTextEncoder":
        self.vec.fit(texts)
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        return self.vec.transform(texts).astype(np.float32)


class Word2VecTextEncoder(BaseTextEncoder):
    def __init__(self, vector_size: int = 128, window: int = 5, min_count: int = 1, epochs: int = 30) -> None:
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.model = None

    def fit(self, texts: List[str]) -> "Word2VecTextEncoder":
        try:
            from gensim.models import Word2Vec
        except Exception as exc:
            raise ImportError("Word2VecTextEncoder requires `gensim`. Install it to use `word2vec`.") from exc

        tokens = [t.split() for t in texts]
        self.model = Word2Vec(
            sentences=tokens,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=1,
            epochs=self.epochs,
        )
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Word2VecTextEncoder must be fitted before transform.")
        out = np.zeros((len(texts), self.vector_size), dtype=np.float32)
        for i, text in enumerate(texts):
            toks = text.split()
            vecs = [self.model.wv[w] for w in toks if w in self.model.wv]
            if vecs:
                out[i] = np.mean(np.stack(vecs), axis=0)
        return out


class PretrainedKeyedVectorsTextEncoder(BaseTextEncoder):
    def __init__(self, kv_path: str = "out/skip_gram_modelling/skip_gram_vectors.kv") -> None:
        self.kv_path = kv_path
        self.kv = None
        self.vector_size = None

    def fit(self, texts: List[str]) -> "PretrainedKeyedVectorsTextEncoder":
        try:
            from gensim.models import KeyedVectors
        except Exception as exc:
            raise ImportError(
                "PretrainedKeyedVectorsTextEncoder requires `gensim`. Install it to use `custom_w2v`."
            ) from exc

        kv_file = Path(self.kv_path)
        if not kv_file.exists():
            raise FileNotFoundError(f"KeyedVectors file not found: {kv_file}")

        self.kv = KeyedVectors.load(str(kv_file), mmap="r")
        self.vector_size = int(self.kv.vector_size)
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        
        if self.kv is None or self.vector_size is None:
            raise RuntimeError("PretrainedKeyedVectorsTextEncoder must be fitted before transform.")
        out = np.zeros((len(texts), self.vector_size), dtype=np.float32)
        for i, text in enumerate(texts):
            toks = text.split()
            vecs = [self.kv[w] for w in toks if w in self.kv]
            if vecs:
                out[i] = np.mean(np.stack(vecs), axis=0)
        return out


class BertEmbeddingEncoder(BaseTextEncoder):
    def __init__(self, model_name: str = "bert-base-uncased", batch_size: int = 16, max_length: int = 256) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, texts: List[str]) -> "BertEmbeddingEncoder":
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("BertEmbeddingEncoder must be fitted before transform.")

        chunks: List[np.ndarray] = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), self.batch_size), desc="Encoding with BERT", unit="batch"):
                batch = texts[i : i + self.batch_size]
                enc = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                ).to(self.device)
                out = self.model(**enc)
                cls = out.last_hidden_state[:, 0, :].detach().cpu().numpy().astype(np.float32)
                chunks.append(cls)

        if not chunks:
            return np.zeros((0, 768), dtype=np.float32)
        return np.concatenate(chunks, axis=0)


def get_encoder(name: str) -> BaseTextEncoder:
    print("Getting encoder for:", name)
    n = name.lower()
    if n == "onehot":
        return OneHotTextEncoder()
    if n in {"bow", "bag_of_words", "count"}:
        return BoWTextEncoder()
    if n in {"tfidf", "tf-idf"}:
        return TfidfTextEncoder()
    if n in {"word2vec", "w2v"}:
        return Word2VecTextEncoder()
    if n.startswith("custom_w2v:") or n.startswith("pretrained_w2v:"):
        _, path = name.split(":", 1)
        return PretrainedKeyedVectorsTextEncoder(kv_path=path)
    if n in {"custom_w2v", "pretrained_w2v", "skipgram_kv", "skip_gram_kv"}:
        return PretrainedKeyedVectorsTextEncoder()
    if n in {"bert", "bert-embed", "bert_embedding"}:
        return BertEmbeddingEncoder()
    raise ValueError(f"Unsupported encoder: {name}")
