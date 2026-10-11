import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import List

from ..config import HTTP_TIMEOUT, USER_AGENT
from ..models import Paper
from .base import Source, SourceError, flatten

API = 'https://export.arxiv.org/api/query'
NS = {'a': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}


class ArxivSource(Source):
    name = 'arxiv'

    def search(self, query: str, limit: int = 10) -> List[Paper]:
        params = {
            'search_query': 'all:' + query,
            'start': '0',
            'max_results': str(limit),
            'sortBy': 'relevance',
            'sortOrder': 'descending',
        }
        url = API + '?' + urllib.parse.urlencode(params)
        request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
                raw = response.read()
        except Exception as exc:
            raise SourceError('arxiv request failed: ' + repr(exc)) from exc
        return self.parse(raw)

    def parse(self, raw) -> List[Paper]:
        if isinstance(raw, bytes):
            raw = raw.decode('utf-8', errors='replace')
        root = ET.fromstring(raw)
        papers = []
        for entry in root.findall('a:entry', NS):
            link = (entry.findtext('a:id', default='', namespaces=NS) or '').strip()
            short_id = link.rstrip('/').rsplit('/', 1)[-1]
            published = entry.findtext('a:published', default='', namespaces=NS) or ''
            year = int(published[:4]) if published[:4].isdigit() else None
            authors = []
            for node in entry.findall('a:author', NS):
                name = flatten(node.findtext('a:name', default='', namespaces=NS))
                if name:
                    authors.append(name)
            doi = flatten(entry.findtext('arxiv:doi', default='', namespaces=NS)) or None
            papers.append(Paper(
                source='arxiv',
                source_id=short_id,
                title=flatten(entry.findtext('a:title', default='', namespaces=NS)),
                abstract=flatten(entry.findtext('a:summary', default='', namespaces=NS)) or None,
                authors=authors,
                year=year,
                venue='arXiv',
                doi=doi,
                url=link or None,
            ))
        return papers
