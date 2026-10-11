from typing import Dict, List

from ..index.embedder import Embedder
from ..index.store import VectorStore
from ..llm import LLMClient
from .citations import resolve_citations
from .prompt import build_messages

DEFAULT_TOP_K = 5
EMPTY_CITATIONS = {'used': [], 'valid': [], 'invalid': [], 'claims': []}


def retrieve(question: str, store: VectorStore, embedder: Embedder,
             top_k: int = DEFAULT_TOP_K) -> List[Dict]:
    vector = embedder.embed_one(question)
    contexts = []
    for position, (score, record) in enumerate(store.search(vector, top_k), 1):
        contexts.append({'n': position, 'score': round(float(score), 4), 'record': record})
    return contexts


def ask(question: str, store: VectorStore, embedder: Embedder, llm: LLMClient,
        top_k: int = DEFAULT_TOP_K, temperature: float = 0.2,
        max_tokens: int = 900) -> Dict:
    contexts = retrieve(question, store, embedder, top_k)
    if not contexts:
        return {
            'question': question,
            'answer': '',
            'contexts': [],
            'citations': dict(EMPTY_CITATIONS),
            'usage': {},
            'empty': True,
        }
    messages = build_messages(question, contexts)
    response = llm.chat(messages, temperature=temperature, max_tokens=max_tokens)
    return {
        'question': question,
        'answer': response.text,
        'contexts': contexts,
        'citations': resolve_citations(response.text, contexts),
        'usage': response.to_dict(),
        'empty': False,
    }
