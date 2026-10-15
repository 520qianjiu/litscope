"""Cross-encoder reranking.

Vector search compares query and passage separately, so it cannot distinguish a
passage that merely shares vocabulary from one that actually answers the question.
A cross-encoder reads the pair together and scores relevance directly, which is the
standard second stage after recall-oriented retrieval.
"""

import json
import os
import time
import urllib.request
from typing import List, Sequence, Tuple

from ..config import HTTP_TIMEOUT

DEFAULT_RERANK_MODEL = 'BAAI/bge-reranker-v2-m3'


class Reranker:
    name = 'none'

    def rerank(self, query: str, documents: Sequence[str],
               top_n: int = 0) -> List[Tuple[int, float]]:
        raise NotImplementedError


class NoopReranker(Reranker):
    name = 'noop'

    def rerank(self, query: str, documents: Sequence[str], top_n: int = 0):
        return [(index, 1.0 / (index + 1)) for index in range(len(documents))]


class OpenAICompatReranker(Reranker):
    name = 'openai-compatible-rerank'

    def __init__(self, base_url: str, api_key: str, model: str,
                 timeout: float = 0.0, attempts: int = 3):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.timeout = float(timeout or HTTP_TIMEOUT)
        self.attempts = max(1, int(attempts))
        self.last_error = ''

    def rerank(self, query: str, documents: Sequence[str], top_n: int = 0):
        items = list(documents)
        if not items:
            return []
        payload = {'model': self.model, 'query': query, 'documents': items}
        if top_n:
            payload['top_n'] = int(top_n)
        body = json.dumps(payload).encode('utf-8')
        last_error = 'unknown'
        for attempt in range(self.attempts):
            request = urllib.request.Request(
                self.base_url + '/rerank', data=body,
                headers={'Content-Type': 'application/json',
                         'Authorization': 'Bearer ' + self.api_key},
                method='POST')
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode('utf-8', errors='replace'))
                rows = data.get('results') or data.get('data') or []
                scored = [(int(row.get('index', 0)), float(row.get('relevance_score', 0.0)))
                          for row in rows]
                scored.sort(key=lambda pair: -pair[1])
                self.last_error = ''
                return scored
            except Exception as exc:
                last_error = type(exc).__name__
                if attempt + 1 < self.attempts:
                    time.sleep(1.5 * (attempt + 1))
        self.last_error = last_error
        # A reranker outage must degrade to the incoming order, never to an empty result.
        return NoopReranker().rerank(query, items, top_n)


def build_reranker(name: str = '') -> Reranker:
    provider = (name or os.environ.get('RERANK_PROVIDER') or 'none').strip().lower()
    if provider in ('', 'none', 'off', 'noop'):
        return NoopReranker()
    if provider in ('openai', 'openai-compatible', 'siliconflow'):
        base_url = (os.environ.get('RERANK_BASE_URL')
                    or os.environ.get('EMBEDDING_BASE_URL') or '').strip()
        api_key = (os.environ.get('RERANK_API_KEY')
                   or os.environ.get('EMBEDDING_API_KEY') or '').strip()
        model = (os.environ.get('RERANK_MODEL') or DEFAULT_RERANK_MODEL).strip()
        missing = [name for name, value in (('RERANK_BASE_URL', base_url),
                                            ('RERANK_API_KEY', api_key),
                                            ('RERANK_MODEL', model)) if not value]
        if missing:
            raise ValueError('rerank provider ' + provider + ' needs: ' + ', '.join(missing))
        return OpenAICompatReranker(base_url, api_key, model)
    raise ValueError('unknown rerank provider: ' + provider)
