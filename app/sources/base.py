from abc import ABC, abstractmethod
from typing import List

from ..models import Paper


class SourceError(RuntimeError):
    pass


def flatten(value) -> str:
    if not value:
        return ''
    return ' '.join(str(value).split())


class Source(ABC):
    name = 'base'

    @abstractmethod
    def search(self, query: str, limit: int = 10) -> List[Paper]:
        raise NotImplementedError
