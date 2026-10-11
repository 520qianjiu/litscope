from typing import Dict, List, Sequence, Tuple

from .bm25 import BM25Index
from .embedder import Embedder
from .store import VectorStore

DEFAULT_TOP_K = 5
Hit = Tuple[float, Dict]


class Retriever:
    name = 'base'

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Hit]:
        raise NotImplementedError


class DenseRetriever(Retriever):
    def __init__(self, store: VectorStore, embedder: Embedder, name: str = ''):
        self.store = store
        self.embedder = embedder
        self.name = name or ('dense-' + embedder.name)

    def __len__(self) -> int:
        return len(self.store)

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Hit]:
        return self.store.search(self.embedder.embed_one(query), top_k)


class BM25Retriever(Retriever):
    name = 'bm25'

    def __init__(self, records: Sequence[Dict], k1: float = 1.5, b: float = 0.75,
                 min_score: float = 0.0):
        self.index = BM25Index(records, k1=k1, b=b)
        self.min_score = float(min_score)

    @classmethod
    def from_store(cls, store: VectorStore, k1: float = 1.5, b: float = 0.75,
                   min_score: float = 0.0) -> 'BM25Retriever':
        return cls(store.records, k1=k1, b=b, min_score=min_score)

    def __len__(self) -> int:
        return len(self.index)

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Hit]:
        return self.index.search(query, top_k, min_score=self.min_score)


def build_retriever(kind: str, store: VectorStore, embedder: Embedder) -> Retriever:
    key = (kind or '').strip().lower()
    if key in ('bm25', 'lexical'):
        return BM25Retriever.from_store(store)
    if key in ('dense', 'embedding', 'vector'):
        return DenseRetriever(store, embedder)
    raise ValueError('unknown retriever: ' + kind)
