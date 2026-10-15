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


def build_fulltext_index(papers, embedder: Embedder, fulltexts,
                         max_chars: int = 480, overlap_sentences: int = 1):
    """Index section-aware full-text chunks.

    Returns (store, stats). Papers whose full text could not be fetched or parsed
    are reported in stats['skipped'] instead of silently vanishing, so a caller can
    tell 'this paper has no full text' from 'this paper was never tried'.
    """
    from .chunker import fulltext_to_chunks

    chunks = []
    skipped = []
    for paper in papers:
        fulltext = fulltexts.get(paper.uid)
        if fulltext is None:
            skipped.append({'paper_uid': paper.uid, 'reason': 'not fetched'})
            continue
        if not getattr(fulltext, 'ok', False):
            skipped.append({'paper_uid': paper.uid,
                            'reason': getattr(fulltext, 'error', '') or 'empty'})
            continue
        produced = fulltext_to_chunks(paper, fulltext, max_chars, overlap_sentences)
        if not produced:
            skipped.append({'paper_uid': paper.uid, 'reason': 'no chunks'})
            continue
        chunks.extend(produced)

    store = VectorStore()
    if chunks:
        vectors = embedder.embed([chunk.text for chunk in chunks])
        records = [{'paper_uid': c.paper_uid, 'chunk_index': c.chunk_index, 'uid': c.uid,
                    'text': c.text, 'section': c.section} for c in chunks]
        store.add(vectors, records)
    stats = {
        'papers_with_fulltext': len(papers) - len(skipped),
        'chunks': len(chunks),
        'sections': sorted({c.section for c in chunks if c.section}),
        'skipped': skipped,
    }
    return store, stats
