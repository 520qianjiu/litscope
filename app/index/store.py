import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

DEFAULT_TOP_K = 5


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right):
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))


class VectorStore:
    def __init__(self) -> None:
        self._vectors: List[List[float]] = []
        self._records: List[Dict] = []

    def __len__(self) -> int:
        return len(self._vectors)

    @property
    def records(self) -> List[Dict]:
        return list(self._records)

    def add(self, vectors: Iterable[Sequence[float]], records: Iterable[Dict]) -> int:
        added = 0
        for vector, record in zip(vectors, records):
            self._vectors.append([float(value) for value in vector])
            self._records.append(dict(record))
            added += 1
        return added

    def search(self, vector: Sequence[float], top_k: int = DEFAULT_TOP_K) -> List[Tuple[float, Dict]]:
        scored = [(cosine(vector, stored), index) for index, stored in enumerate(self._vectors)]
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        picked = scored[:max(0, int(top_k))]
        return [(score, dict(self._records[index])) for score, index in picked]

    def save(self, path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {'vectors': self._vectors, 'records': self._records}
        target.write_text(json.dumps(payload), encoding='utf-8')

    @classmethod
    def load(cls, path) -> 'VectorStore':
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        store = cls()
        store.add(payload.get('vectors') or [], payload.get('records') or [])
        return store
