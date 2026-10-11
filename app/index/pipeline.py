from typing import Dict, List, Sequence

from ..models import Paper
from .chunker import paper_to_chunks
from .embedder import Embedder
from .store import VectorStore


def build_index(papers: Sequence[Paper], embedder: Embedder,
                max_chars: int = 480, overlap_sentences: int = 1) -> VectorStore:
    chunks = []
    for paper in papers:
        chunks.extend(paper_to_chunks(paper, max_chars, overlap_sentences))
    store = VectorStore()
    if not chunks:
        return store
    vectors = embedder.embed([chunk.text for chunk in chunks])
    records: List[Dict] = []
    for chunk in chunks:
        records.append({
            'paper_uid': chunk.paper_uid,
            'chunk_index': chunk.chunk_index,
            'uid': chunk.uid,
            'text': chunk.text,
        })
    store.add(vectors, records)
    return store
