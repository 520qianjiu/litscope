from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class Paper:
    source: str
    source_id: str
    title: str
    abstract: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    citations: Optional[int] = None
    merged_sources: List[str] = field(default_factory=list)

    @property
    def uid(self) -> str:
        return self.source + ':' + self.source_id

    def to_dict(self) -> dict:
        data = asdict(self)
        data['uid'] = self.uid
        return data


@dataclass
class Chunk:
    paper_uid: str
    chunk_index: int
    text: str
    embedding: Optional[List[float]] = None
    section: str = ''

    @property
    def uid(self) -> str:
        return self.paper_uid + '#' + str(self.chunk_index)


@dataclass
class Citation:
    paper_uid: str
    chunk_index: int
    quote: str
