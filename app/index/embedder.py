import hashlib
import json
import math
import os
import urllib.request
from typing import List, Sequence

from ..config import HTTP_TIMEOUT


class Embedder:
    name = 'base'
    dim = 0

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]


class HashingEmbedder(Embedder):
    name = 'hashing'

    def __init__(self, dim: int = 256, ngram: int = 3):
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

    def __init__(self, base_url: str, api_key: str, model: str, dim: int = 0):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.dim = int(dim)

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        payload = {'model': self.model, 'input': list(texts)}
        request = urllib.request.Request(
            self.base_url + '/embeddings',
            data=json.dumps(payload).encode('utf-8'),
            headers={
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + self.api_key,
            },
            method='POST',
        )
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
            data = json.loads(response.read().decode('utf-8', errors='replace'))
        rows = sorted(data.get('data') or [], key=lambda row: row.get('index', 0))
        vectors = [[float(value) for value in row.get('embedding') or []] for row in rows]
        if vectors and not self.dim:
            self.dim = len(vectors[0])
        return vectors


def build_embedder(name: str = '') -> Embedder:
    provider = (name or os.environ.get('EMBEDDING_PROVIDER') or 'hashing').strip().lower()
    if provider in ('hashing', 'local', 'offline'):
        return HashingEmbedder(
            dim=int(os.environ.get('EMBEDDING_DIM') or 256),
            ngram=int(os.environ.get('EMBEDDING_NGRAM') or 3),
        )
    if provider in ('openai', 'openai-compatible', 'siliconflow', 'zhipu'):
        base_url = os.environ.get('EMBEDDING_BASE_URL', '')
        api_key = os.environ.get('EMBEDDING_API_KEY', '')
        model = os.environ.get('EMBEDDING_MODEL', '')
        if not (base_url and api_key and model):
            raise ValueError('EMBEDDING_BASE_URL / EMBEDDING_API_KEY / EMBEDDING_MODEL are required')
        return OpenAICompatEmbedder(base_url, api_key, model,
                                    dim=int(os.environ.get('EMBEDDING_DIM') or 0))
    raise ValueError('unknown embedding provider: ' + provider)
