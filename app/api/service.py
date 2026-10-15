import os
import tempfile
import time
from typing import Dict, List, Optional

from ..dedupe import dedupe
from ..fulltext import load_many
from ..index import HashingEmbedder, build_embedder
from ..index.pipeline import build_fulltext_index, build_index
from ..index.rerank import build_reranker
from ..index.retrievers import RerankedRetriever, TwoStageRetriever, build_retriever
from ..llm import LLMClient, build_llm
from ..rag import ask_with_retriever
from ..sources import build_sources

DEFAULT_SOURCES = ['arxiv', 'crossref', 'pubmed']
DEFAULT_RETRIEVER = (os.environ.get('LITSCOPE_RETRIEVER') or 'bm25').strip().lower()
FULLTEXT_RETRIEVER = 'fulltext'
RERANK_CANDIDATES = int(os.environ.get('RERANK_CANDIDATES') or 20)
# Full text is only ever used by the second stage, which looks at a handful of
# shortlisted papers. Fetching bodies for the whole corpus multiplies first-request
# latency (PDF download + embedding) for no recall gain, so it is capped.
FULLTEXT_PAPERS = int(os.environ.get('LITSCOPE_FULLTEXT_PAPERS') or 6)
SECRET_KEYS = {'DEEPSEEK_API_KEY', 'EMBEDDING_API_KEY'}
CONFIG_KEYS = [
    'LLM_PROVIDER',
    'DEEPSEEK_API_KEY',
    'DEEPSEEK_BASE_URL',
    'DEEPSEEK_MODEL',
    'EMBEDDING_PROVIDER',
    'EMBEDDING_BASE_URL',
    'EMBEDDING_API_KEY',
    'EMBEDDING_MODEL',
    'EMBEDDING_DIM',
    'EMBEDDING_NGRAM',
    'LITSCOPE_RETRIEVER',
    'LITSCOPE_FULLTEXT_PAPERS',
    'LITSCOPE_FULLTEXT_CONCURRENCY',
    'EMBEDDING_BATCH',
    'EMBEDDING_CONCURRENCY',
]


class CorpusService:
    def __init__(self, llm: Optional[LLMClient] = None, embedder=None,
                 source_factory=None, retriever: str = DEFAULT_RETRIEVER,
                 fulltext_loader=None, reranker=None, fulltext_cache: str = ''):
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
        self.fulltext_loader = fulltext_loader or load_many
        # Reranking is optional: if it is not configured the retriever simply falls
        # back to the incoming order rather than failing.
        try:
            self.reranker = reranker if reranker is not None else build_reranker()
        except Exception as exc:
            self.reranker = build_reranker('none')
            self.embedder_error = (self.embedder_error + ' | rerank: ' + str(exc)[:160]).strip(' |')
        self.fulltext_cache = fulltext_cache or os.environ.get('LITSCOPE_FULLTEXT_CACHE') or os.path.join(
            tempfile.gettempdir(), 'litscope-fulltext')

    def config_report(self) -> Dict[str, str]:
        # Non-secret values are shown verbatim so a misconfigured deployment can be
        # diagnosed from the outside. Secrets only ever report their presence.
        report = {}
        for key in CONFIG_KEYS:
            value = (os.environ.get(key) or '').strip()
            if not value:
                report[key] = 'MISSING'
            elif key in SECRET_KEYS:
                report[key] = 'set(len=' + str(len(value)) + ')'
            else:
                report[key] = value[:60]
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

        started = time.monotonic()
        pooled = []
        errors = []
        for source in self.source_factory(chosen):
            try:
                pooled.extend(source.search(topic, per_source))
            except Exception as exc:
                errors.append({'source': source.name, 'error': str(exc)[:200]})
        searched = time.monotonic()

        papers = dedupe(pooled)
        store = build_index(papers, self.embedder)
        indexed = time.monotonic()
        entry = {
            'topic': topic,
            'papers': papers,
            'store': store,
            'errors': errors,
            'sources': chosen,
            'retrievers': {},
            'fulltext_store': None,
            'fulltext': None,
            # Stage timings travel with the API response so a cold start can be
            # attributed to one stage instead of guessed at.
            'timings': {'sources': round(searched - started, 2),
                        'abstract_index': round(indexed - searched, 2)},
        }
        self.cache[key] = entry

        if self.wants_fulltext():
            entry['fulltext'] = self.load_fulltext(entry)
        entry['timings']['build_total'] = round(time.monotonic() - started, 2)
        return entry

    def wants_fulltext(self) -> bool:
        return (self.default_retriever == FULLTEXT_RETRIEVER
                or (os.environ.get('LITSCOPE_FULLTEXT') or '').strip().lower()
                in ('1', 'true', 'yes', 'on'))

    def load_fulltext(self, entry: dict) -> dict:
        """Fetch bodies for the corpus and index them. Never raises: a topic where no
        full text is available simply keeps using the abstract index."""
        papers = entry['papers']
        timings = entry.setdefault('timings', {})
        summary = {'papers': len(papers), 'with_fulltext': 0, 'chunks': 0, 'error': ''}
        started = time.monotonic()
        try:
            fetched = self.fulltext_loader(papers[:FULLTEXT_PAPERS], cache_dir=self.fulltext_cache)
        except Exception as exc:
            summary['error'] = type(exc).__name__
            timings['fulltext_fetch'] = round(time.monotonic() - started, 2)
            return summary
        timings['fulltext_fetch'] = round(time.monotonic() - started, 2)
        usable = sum(1 for value in fetched.values() if getattr(value, 'ok', False))
        summary['with_fulltext'] = usable
        if not usable:
            # Surface *why* nothing came back: 'arxiv blocked the IP' and 'this topic
            # has no open-access papers' need very different responses.
            summary['error'] = 'no full text available'
            summary['reasons'] = [
                {'paper': uid, 'reason': (getattr(value, 'error', '') or 'unknown')[:90]}
                for uid, value in list(fetched.items())[:4]
            ]
            return summary
        started = time.monotonic()
        try:
            store, stats = build_fulltext_index(papers, self.embedder, fetched)
        except Exception as exc:
            summary['error'] = type(exc).__name__
            timings['fulltext_index'] = round(time.monotonic() - started, 2)
            return summary
        timings['fulltext_index'] = round(time.monotonic() - started, 2)
        entry['fulltext_store'] = store if len(store) else None
        summary['chunks'] = stats['chunks']
        summary['sections'] = len(stats['sections'])
        return summary

    def base_retriever_for(self, entry: dict, kind: str):
        name = (kind or self.default_retriever or DEFAULT_RETRIEVER).strip().lower()
        cache = entry.setdefault('retrievers', {})
        if name not in cache:
            if name == FULLTEXT_RETRIEVER and entry.get('fulltext_store') is not None:
                two_stage = TwoStageRetriever(entry['store'], entry['fulltext_store'],
                                              self.embedder, name='two-stage')
                cache[name] = RerankedRetriever(two_stage, self.reranker,
                                                candidates=RERANK_CANDIDATES,
                                                name='fulltext+' + self.reranker.name)
            else:
                fallback = 'dense' if name == FULLTEXT_RETRIEVER else name
                cache[name] = build_retriever(fallback, entry['store'], self.embedder)
        return cache[name]

    def retriever_for(self, entry: dict, kind: str = ''):
        return self.base_retriever_for(entry, kind)

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
        started = time.monotonic()
        entry = self.corpus(topic, per_source, sources)
        corpus_ready = time.monotonic()
        engine = self.retriever_for(entry, retriever)
        result = ask_with_retriever(question, engine, self.llm, top_k=top_k)
        answered = time.monotonic()
        # corpus_wait is ~0 on a cache hit; otherwise it is the full build time.
        result['timings'] = dict(entry.get('timings') or {})
        result['timings'].update({'corpus_wait': round(corpus_ready - started, 2),
                                  'answer': round(answered - corpus_ready, 2)})
        result['corpus'] = {
            'topic': topic,
            'papers': len(entry['papers']),
            'chunks': len(entry['store']),
            'errors': entry['errors'],
            'fulltext': entry.get('fulltext'),
            'reranker': self.reranker.name,
        }
        return result
