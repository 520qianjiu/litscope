import math
import re
from collections import Counter
from typing import Dict, List, Sequence, Tuple

TOKEN = re.compile(r'[0-9a-z\u4e00-\u9fff]+')
DEFAULT_TOP_K = 5


def tokenize(text: str) -> List[str]:
    return TOKEN.findall((text or '').lower())


class BM25Index:
    def __init__(self, records: Sequence[Dict], k1: float = 1.5, b: float = 0.75):
        self.records = [dict(record) for record in records]
        self.k1 = float(k1)
        self.b = float(b)
        self.docs = [tokenize(record.get('text') or '') for record in self.records]
        self.lengths = [len(doc) for doc in self.docs]
        self.avgdl = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        self.term_freqs = [Counter(doc) for doc in self.docs]
        doc_freq: Counter = Counter()
        for terms in self.docs:
            for term in set(terms):
                doc_freq[term] += 1
        total = len(self.docs)
        self.idf = {
            term: math.log(1.0 + (total - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def __len__(self) -> int:
        return len(self.records)

    def score(self, query: str, index: int) -> float:
        terms = tokenize(query)
        if not terms:
            return 0.0
        counter = self.term_freqs[index]
        length = self.lengths[index] or 1
        total = 0.0
        for term in terms:
            freq = counter.get(term)
            if not freq:
                continue
            idf = self.idf.get(term, 0.0)
            denominator = freq + self.k1 * (1.0 - self.b + self.b * length / (self.avgdl or 1.0))
            total += idf * (freq * (self.k1 + 1.0)) / denominator
        return total

    def search(self, query: str, top_k: int = DEFAULT_TOP_K,
               min_score: float = 0.0) -> List[Tuple[float, Dict]]:
        # A zero score means NO lexical overlap at all. Returning those hits would be
        # worse than returning nothing: the generator would be fed arbitrary chunks.
        # This is what happens when a Chinese query meets an English-only corpus.
        scored = [(self.score(query, i), i) for i in range(len(self.records))]
        scored = [(value, i) for value, i in scored if value > min_score]
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [(value, dict(self.records[i])) for value, i in scored[:max(0, int(top_k))]]
