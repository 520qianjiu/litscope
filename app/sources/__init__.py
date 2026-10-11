from .arxiv import ArxivSource
from .base import Source, SourceError
from .crossref import CrossrefSource
from .pubmed import PubmedSource

REGISTRY = {
    'arxiv': ArxivSource,
    'crossref': CrossrefSource,
    'pubmed': PubmedSource,
}


def build_sources(names=None):
    selected = list(names) if names else list(REGISTRY)
    sources = []
    for name in selected:
        key = name.strip().lower()
        if key not in REGISTRY:
            raise ValueError('unknown source: ' + name)
        sources.append(REGISTRY[key]())
    return sources


__all__ = [
    'Source',
    'SourceError',
    'ArxivSource',
    'CrossrefSource',
    'PubmedSource',
    'REGISTRY',
    'build_sources',
]
