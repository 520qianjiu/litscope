import json
import re
import urllib.parse
import urllib.request
from typing import List

from ..config import HTTP_TIMEOUT, USER_AGENT
from ..models import Paper
from .base import Source, SourceError, flatten

API = 'https://api.crossref.org/works'
SELECT = 'DOI,title,abstract,author,issued,container-title,is-referenced-by-count,URL'
_TAG = re.compile(r'<[^>]+>')


def strip_tags(text: str) -> str:
    return flatten(_TAG.sub(' ', text or ''))


class CrossrefSource(Source):
    name = 'crossref'

    def search(self, query: str, limit: int = 10) -> List[Paper]:
        params = {'query': query, 'rows': str(limit), 'select': SELECT}
        url = API + '?' + urllib.parse.urlencode(params)
        request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
                raw = response.read()
        except Exception as exc:
            raise SourceError('crossref request failed: ' + repr(exc)) from exc
        return self.parse(raw)

    def parse(self, raw) -> List[Paper]:
        if isinstance(raw, bytes):
            raw = raw.decode('utf-8', errors='replace')
        data = json.loads(raw)
        items = (data.get('message') or {}).get('items') or []
        papers = []
        for item in items:
            doi = (item.get('DOI') or '').strip() or None
            titles = item.get('title') or []
            title = flatten(titles[0]) if titles else ''
            authors = []
            for author in item.get('author') or []:
                name = ((author.get('given') or '') + ' ' + (author.get('family') or '')).strip()
                if name:
                    authors.append(name)
            year = None
            parts = (item.get('issued') or {}).get('date-parts') or []
            if parts and parts[0] and isinstance(parts[0][0], int):
                year = parts[0][0]
            venues = item.get('container-title') or []
            papers.append(Paper(
                source='crossref',
                source_id=doi or title,
                title=title,
                abstract=strip_tags(item.get('abstract') or '') or None,
                authors=authors,
                year=year,
                venue=flatten(venues[0]) if venues else None,
                doi=doi,
                url=item.get('URL') or None,
                citations=item.get('is-referenced-by-count'),
            ))
        return papers
