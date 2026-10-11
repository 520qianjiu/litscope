from typing import Dict, List, Optional

from ..dedupe import dedupe
from ..index import build_embedder
from ..index.pipeline import build_index
from ..llm import LLMClient, build_llm
from ..rag import ask as rag_ask
from ..sources import build_sources

DEFAULT_SOURCES = ['arxiv', 'crossref', 'pubmed']


class CorpusService:
    def __init__(self, llm: Optional[LLMClient] = None, embedder=None, source_factory=None):
        self._llm = llm
        self.embedder = embedder or build_embedder()
        self.source_factory = source_factory or build_sources
        self.cache: Dict[str, dict] = {}

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = build_llm()
        return self._llm

    @property
    def llm_ready(self) -> bool:
        try:
            return self.llm is not None
        except Exception:
            return False

    @staticmethod
    def cache_key(topic: str, per_source: int, sources: List[str]) -> str:
        return topic.strip().lower() + '|' + str(per_source) + '|' + ','.join(sources)

    def corpus(self, topic: str, per_source: int = 8, sources: Optional[List[str]] = None) -> dict:
        chosen = list(sources) if sources else list(DEFAULT_SOURCES)
        key = self.cache_key(topic, per_source, chosen)
        if key in self.cache:
            return self.cache[key]

        pooled = []
        errors = []
        for source in self.source_factory(chosen):
            try:
                pooled.extend(source.search(topic, per_source))
            except Exception as exc:
                errors.append({'source': source.name, 'error': str(exc)[:200]})

        papers = dedupe(pooled)
        store = build_index(papers, self.embedder)
        entry = {
            'topic': topic,
            'papers': papers,
            'store': store,
            'errors': errors,
            'sources': chosen,
        }
        self.cache[key] = entry
        return entry

    def search(self, query: str, limit: int = 10, sources: Optional[List[str]] = None) -> dict:
        chosen = list(sources) if sources else list(DEFAULT_SOURCES)
        pooled = []
        errors = []
        for source in self.source_factory(chosen):
            try:
                pooled.extend(source.search(query, limit))
            except Exception as exc:
                errors.append({'source': source.name, 'error': str(exc)[:200]})
        papers = dedupe(pooled)
        return {
            'query': query,
            'raw': len(pooled),
            'count': len(papers),
            'papers': [paper.to_dict() for paper in papers],
            'errors': errors,
        }

    def ask(self, topic: str, question: str, top_k: int = 5,
            per_source: int = 8, sources: Optional[List[str]] = None) -> dict:
        entry = self.corpus(topic, per_source, sources)
        result = rag_ask(question, entry['store'], self.embedder, self.llm, top_k=top_k)
        result['corpus'] = {
            'topic': topic,
            'papers': len(entry['papers']),
            'chunks': len(entry['store']),
            'errors': entry['errors'],
        }
        return result
