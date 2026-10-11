from typing import Dict, List

from ..index.embedder import Embedder
from ..index.retrievers import BM25Retriever, DenseRetriever, Retriever
from ..index.store import VectorStore
from ..llm import LLMClient
from .citations import resolve_citations
from .prompt import build_messages

DEFAULT_TOP_K = 5
EMPTY_CITATIONS = {'used': [], 'valid': [], 'invalid': [], 'claims': []}


def contexts_from_hits(hits) -> List[Dict]:
    return [
        {'n': position, 'score': round(float(score), 4), 'record': record}
        for position, (score, record) in enumerate(hits, 1)
    ]


def retrieve_with(retriever: Retriever, question: str, top_k: int = DEFAULT_TOP_K) -> List[Dict]:
    return contexts_from_hits(retriever.search(question, top_k))


def retrieve(question: str, store: VectorStore, embedder: Embedder,
             top_k: int = DEFAULT_TOP_K) -> List[Dict]:
    return retrieve_with(DenseRetriever(store, embedder), question, top_k)


def _answer(question: str, contexts: List[Dict], llm: LLMClient, retriever_name: str,
            temperature: float, max_tokens: int) -> Dict:
    if not contexts:
        return {
            'question': question,
            'answer': '',
            'contexts': [],
            'citations': dict(EMPTY_CITATIONS),
            'usage': {},
            'retriever': retriever_name,
            'empty': True,
        }
    response = llm.chat(build_messages(question, contexts), temperature=temperature,
                        max_tokens=max_tokens)
    return {
        'question': question,
        'answer': response.text,
        'contexts': contexts,
        'citations': resolve_citations(response.text, contexts),
        'usage': response.to_dict(),
        'retriever': retriever_name,
        'empty': False,
    }


def ask_with_retriever(question: str, retriever: Retriever, llm: LLMClient,
                       top_k: int = DEFAULT_TOP_K, temperature: float = 0.2,
                       max_tokens: int = 900) -> Dict:
    contexts = retrieve_with(retriever, question, top_k)
    return _answer(question, contexts, llm, retriever.name, temperature, max_tokens)


def ask(question: str, store: VectorStore, embedder: Embedder, llm: LLMClient,
        top_k: int = DEFAULT_TOP_K, temperature: float = 0.2,
        max_tokens: int = 900) -> Dict:
    return ask_with_retriever(question, DenseRetriever(store, embedder), llm,
                              top_k=top_k, temperature=temperature, max_tokens=max_tokens)


def build_retrievers(store: VectorStore, embedder: Embedder, kinds) -> List[Retriever]:
    out: List[Retriever] = []
    for kind in kinds:
        key = (kind or '').strip().lower()
        if key == 'bm25':
            out.append(BM25Retriever.from_store(store))
        else:
            out.append(DenseRetriever(store, embedder))
    return out
