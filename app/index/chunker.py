import re
from typing import List

from ..models import Chunk, Paper

_SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])\s+')
DEFAULT_MAX_CHARS = 480
DEFAULT_OVERLAP = 1


def split_sentences(text: str) -> List[str]:
    cleaned = (text or '').strip()
    if not cleaned:
        return []
    return [part.strip() for part in _SENTENCE_BOUNDARY.split(cleaned) if part.strip()]


def chunk_text(text: str, max_chars: int = DEFAULT_MAX_CHARS,
               overlap_sentences: int = DEFAULT_OVERLAP) -> List[str]:
    sentences = split_sentences(text)
    if not sentences:
        return []
    chunks: List[str] = []
    current: List[str] = []
    size = 0
    for sentence in sentences:
        projected = size + len(sentence) + 1
        if current and projected > max_chars:
            chunks.append(' '.join(current))
            keep = current[-overlap_sentences:] if overlap_sentences > 0 else []
            current = list(keep)
            size = sum(len(item) + 1 for item in current)
        current.append(sentence)
        size += len(sentence) + 1
    if current:
        chunks.append(' '.join(current))
    return chunks


def paper_to_chunks(paper: Paper, max_chars: int = DEFAULT_MAX_CHARS,
                    overlap_sentences: int = DEFAULT_OVERLAP) -> List[Chunk]:
    body = ((paper.title or '').strip() + '. ' + (paper.abstract or '').strip()).strip()
    pieces = chunk_text(body, max_chars, overlap_sentences)
    return [Chunk(paper_uid=paper.uid, chunk_index=index, text=piece)
            for index, piece in enumerate(pieces)]


def sections_to_chunks(paper_uid: str, sections, max_chars: int = DEFAULT_MAX_CHARS,
                       overlap_sentences: int = DEFAULT_OVERLAP) -> List[Chunk]:
    """Chunk a parsed full text, keeping the section label on every chunk.

    The label is what lets retrieval distinguish a claim made in RESULTS from a
    sentence in RELATED WORK, which matters because the abstract cannot tell the
    two apart at all.
    """
    chunks: List[Chunk] = []
    for section in sections:
        name = (getattr(section, 'name', '') or '').strip()
        pieces = chunk_text(getattr(section, 'text', ''), max_chars, overlap_sentences)
        for piece in pieces:
            chunks.append(Chunk(paper_uid=paper_uid, chunk_index=len(chunks),
                                text=piece, section=name))
    return chunks


def fulltext_to_chunks(paper, fulltext, max_chars: int = DEFAULT_MAX_CHARS,
                       overlap_sentences: int = DEFAULT_OVERLAP) -> List[Chunk]:
    return sections_to_chunks(paper.uid, fulltext.sections, max_chars, overlap_sentences)
