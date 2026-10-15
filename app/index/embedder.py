import hashlib
import json
import math
import os
import urllib.request
from typing import List, Sequence

from ..config import HTTP_TIMEOUT

DEFAULT_HASHING_DIM = 256
DEFAULT_HASHING_NGRAM = 3


def _as_int(name: str, default: int) -> int:
    # A malformed numeric env var must never take the service down. Fall back to the
    # default instead of raising: this exact crash happened when EMBEDDING_DIM held a
    # model name (BAAI/bge-m3) and int() blew up at import time.
    raw = (os.environ.get(name) or '').strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


class Embedder:
    name = 'base'
    dim = 0

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]


class HashingEmbedder(Embedder):
    name = 'hashing'

    def __init__(self, dim: int = DEFAULT_HASHING_DIM, ngram: int = DEFAULT_HASHING_NGRAM):
        self.dim = int(dim)
        self.ngram = int(ngram)

    def _tokens(self, text: str) -> List[str]:
        cleaned = ''.join(ch.lower() if ch.isalnum() else ' ' for ch in (text or ''))
        words = [w for w in cleaned.split() if w]
        tokens = list(words)
        for word in words:
            if len(word) <= self.ngram:
                tokens.append(word)
                continue
            for start in range(len(word) - self.ngram + 1):
                tokens.append(word[start:start + self.ngram])
        return tokens

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.dim
            for token in self._tokens(text):
                digest = hashlib.md5(token.encode('utf-8')).digest()
                slot = int.from_bytes(digest[:4], 'big') % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vector[slot] += sign
            norm = math.sqrt(sum(value * value for value in vector))
            if norm > 0:
                vector = [value / norm for value in vector]
            vectors.append(vector)
        return vectors


class OpenAICompatEmbedder(Embedder):
    name = 'openai-compatible'

    def __init__(self, base_url: str, api_key: str, model: str, dim: int = 0,
                 batch_size: int = 64, timeout: float = 0):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.dim = int(dim)
        self.batch_size = max(1, int(batch_size))
        self.timeout = float(timeout or HTTP_TIMEOUT)

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        # Full-text indexing produces thousands of chunks; a single request would
        # time out, so requests are batched. The batch size is configurable because
        # providers differ a lot in how much they accept per call.
        items = list(texts)
        if not items:
            return []
        size = max(1, int(os.environ.get('EMBEDDING_BATCH') or self.batch_size))
        vectors: List[List[float]] = []
        for start in range(0, len(items), size):
            vectors.extend(self._embed_batch(items[start:start + size]))
        return vectors

    def _embed_batch(self, batch: Sequence[str], attempts: int = 3) -> List[List[float]]:
        payload = {'model': self.model, 'input': list(batch)}
        body = json.dumps(payload).encode('utf-8')
        last_error = 'unknown'
        for attempt in range(attempts):
            request = urllib.request.Request(
                self.base_url + '/embeddings',
                data=body,
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer ' + self.api_key,
                },
                method='POST',
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode('utf-8', errors='replace'))
                rows = sorted(data.get('data') or [], key=lambda row: row.get('index', 0))
                vectors = [[float(v) for v in row.get('embedding') or []] for row in rows]
                if vectors and not self.dim:
                    self.dim = len(vectors[0])
                return vectors
            except Exception as exc:
                last_error = type(exc).__name__
                if attempt + 1 < attempts:
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError('embedding request failed after %d attempts: %s'
                           % (attempts, last_error))


def build_embedder(name: str = '') -> Embedder:
    provider = (name or os.environ.get('EMBEDDING_PROVIDER') or 'hashing').strip().lower()
    if provider in ('hashing', 'local', 'offline'):
        return HashingEmbedder(
            dim=_as_int('EMBEDDING_DIM', DEFAULT_HASHING_DIM),
            ngram=_as_int('EMBEDDING_NGRAM', DEFAULT_HASHING_NGRAM),
        )
    if provider in ('openai', 'openai-compatible', 'siliconflow', 'zhipu'):
        base_url = (os.environ.get('EMBEDDING_BASE_URL') or '').strip()
        api_key = (os.environ.get('EMBEDDING_API_KEY') or '').strip()
        model = (os.environ.get('EMBEDDING_MODEL') or '').strip()
        missing = [key for key, value in (
            ('EMBEDDING_BASE_URL', base_url),
            ('EMBEDDING_API_KEY', api_key),
            ('EMBEDDING_MODEL', model),
        ) if not value]
        if missing:
            raise ValueError('embedding provider ' + provider + ' needs: ' + ', '.join(missing))
        return OpenAICompatEmbedder(base_url, api_key, model, dim=_as_int('EMBEDDING_DIM', 0))
    raise ValueError('unknown embedding provider: ' + provider)
