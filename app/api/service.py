import os
from typing import Dict, List, Optional

from ..dedupe import dedupe
from ..index import HashingEmbedder, build_embedder
from ..index.pipeline import build_index
from ..index.retrievers import build_retriever
from ..llm import LLMClient, build_llm
from ..rag import ask_with_retriever
from ..sources import build_sources

DEFAULT_SOURCES = ['arxiv', 'crossref', 'pubmed']
DEFAULT_RETRIEVER = (os.environ.get('LITSCOPE_RETRIEVER') or 'bm25').strip().lower()
CONFIG_KEYS = [
    'LLM_PROVIDER',
    'DEEPSEEK_API_KEY',
    'DEEPSEEK_BASE_URL',
    'DEEPSEEK_MODEL',
    'EMBEDDING_PROVIDER',
    'EMBEDDING_BASE_URL',
    'EMBEDDING_API_KEY',
    'EMBEDDING_MODEL',
    'LITSCOPE_RETRIEVER',
]


class CorpusService:
    def __init__(self, llm: Optional[LLMClient] = None, embedder=None,
                 source_factory=None, retriever: str = DEFAULT_RETRIEVER):
        self._llm = llm
        self.embedder_error = ''
        if embedder is not None:
            self.embedder = embedder
        else:
            try:
                self.embedder = build_embedder()
            except Exception as exc:
                # A config typo must not take the whole service down. Start with the
                # offline fallback and surface the real reason via /api/health.
                self.embedder_error = str(exc)[:300]
                try:
                    self.embedder = build_embedder('hashing')
                except Exception:
                    # Belt and braces: the fallback itself must never take the service down.
                    self.embedder = HashingEmbedder()
        self.source_factory = source_factory or build_sources
        self.default_retriever = retriever
        self.cache: Dict[str, dict] = {}

    def config_report(self) -> Dict[str, str]:
        report = {}
        for key in CONFIG_KEYS:
            report[key] = 'set' if (os.environ.get(key) or '').strip() else 'MISSING'
        return report

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
            'retrievers': {},
        }
        self.cache[key] = entry
        return entry

    def retriever_for(self, entry: dict, kind: str = ''):
        name = (kind or self.default_retriever or DEFAULT_RETRIEVER).strip().lower()
        cache = entry.setdefault('retrievers', {})
        if name not in cache:
            cache[name] = build_retriever(name, entry['store'], self.embedder)
        return cache[name]

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

    def ask(self, topic: str, question: str, top_k: int = 5, per_source: int = 8,
            sources: Optional[List[str]] = None, retriever: str = '') -> dict:
        entry = self.corpus(topic, per_source, sources)
        engine = self.retriever_for(entry, retriever)
        result = ask_with_retriever(question, engine, self.llm, top_k=top_k)
        result['corpus'] = {
            'topic': topic,
            'papers': len(entry['papers']),
            'chunks': len(entry['store']),
            'errors': entry['errors'],
        }
        return result
