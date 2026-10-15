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


class TwoStageRetriever(Retriever):
    """Stage one shortlists papers from abstracts; stage two ranks their full text.

    This exists because full texts do not fit in a prompt: 20 papers is roughly
    600k tokens. Shortlisting first keeps the context small enough to send while
    still letting the answer be grounded in body text instead of abstracts.
    """

    name = 'two-stage'

    def __init__(self, abstract_store: VectorStore, fulltext_store: VectorStore,
                 embedder: Embedder, candidates: int = 5, name: str = ''):
        self.abstract_store = abstract_store
        self.fulltext_store = fulltext_store
        self.embedder = embedder
        self.candidates = max(1, int(candidates))
        self.last_shortlist: List[str] = []
        if name:
            self.name = name

    def __len__(self) -> int:
        return len(self.fulltext_store)

    def shortlist(self, query: str):
        vector = self.embedder.embed_one(query)
        hits = self.abstract_store.search(vector, self.candidates)
        seen = set()
        paper_ids: List[str] = []
        for _score, record in hits:
            uid = record.get('paper_uid')
            if uid and uid not in seen:
                seen.add(uid)
                paper_ids.append(uid)
        self.last_shortlist = paper_ids
        return vector, paper_ids

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Hit]:
        vector, paper_ids = self.shortlist(query)
        if not paper_ids:
            return []
        return self.fulltext_store.search(vector, top_k, paper_ids=paper_ids)
