import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import List

from ..config import HTTP_TIMEOUT, PUBMED_API_KEY, PUBMED_EMAIL, USER_AGENT
from ..models import Paper
from .base import Source, SourceError, flatten

ESEARCH = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
EFETCH = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'


class PubmedSource(Source):
    name = 'pubmed'

    def search(self, query: str, limit: int = 10) -> List[Paper]:
        ids = self._esearch(query, limit)
        if not ids:
            return []
        return self.parse(self._efetch(ids))

    def _common_params(self) -> dict:
        params = {'tool': 'LitScope'}
        if PUBMED_EMAIL:
            params['email'] = PUBMED_EMAIL
        if PUBMED_API_KEY:
            params['api_key'] = PUBMED_API_KEY
        return params

    def _get(self, base: str, params: dict, label: str) -> str:
        url = base + '?' + urllib.parse.urlencode(params)
        request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
                return response.read().decode('utf-8', errors='replace')
        except Exception as exc:
            raise SourceError('pubmed ' + label + ' failed: ' + repr(exc)) from exc

    def _esearch(self, query: str, limit: int) -> List[str]:
        params = self._common_params()
        params.update({'db': 'pubmed', 'term': query, 'retmax': str(limit),
                       'retmode': 'json', 'sort': 'relevance'})
        data = json.loads(self._get(ESEARCH, params, 'esearch'))
        return ((data.get('esearchresult') or {}).get('idlist')) or []

    def _efetch(self, ids: List[str]) -> str:
        params = self._common_params()
        params.update({'db': 'pubmed', 'id': ','.join(ids), 'retmode': 'xml'})
        return self._get(EFETCH, params, 'efetch')

    def parse(self, raw) -> List[Paper]:
        if isinstance(raw, bytes):
            raw = raw.decode('utf-8', errors='replace')
        root = ET.fromstring(raw)
        papers = []
        for article in root.findall('.//PubmedArticle'):
            pmid = flatten(article.findtext('.//PMID', default=''))
            papers.append(Paper(
                source='pubmed',
                source_id=pmid or flatten(article.findtext('.//ArticleTitle', default='')),
                title=flatten(article.findtext('.//ArticleTitle', default='')),
                abstract=self._abstract(article),
                authors=self._authors(article),
                year=self._year(article),
                venue=flatten(article.findtext('.//Journal/Title', default='')) or None,
                doi=self._doi(article),
                url='https://pubmed.ncbi.nlm.nih.gov/' + pmid + '/' if pmid else None,
            ))
        return papers

    @staticmethod
    def _abstract(article) -> str:
        parts = []
        for node in article.findall('.//Abstract/AbstractText'):
            text = flatten(''.join(node.itertext()))
            if not text:
                continue
            label = (node.get('Label') or '').strip()
            parts.append(label + ': ' + text if label else text)
        return ' '.join(parts) or None

    @staticmethod
    def _authors(article) -> List[str]:
        names = []
        for node in article.findall('.//AuthorList/Author'):
            last = flatten(node.findtext('LastName', default=''))
            fore = flatten(node.findtext('ForeName', default=''))
            collective = flatten(node.findtext('CollectiveName', default=''))
            name = (fore + ' ' + last).strip() if (fore or last) else collective
            if name:
                names.append(name)
        return names

    @staticmethod
    def _year(article):
        raw = flatten(article.findtext('.//JournalIssue/PubDate/Year', default=''))
        if raw.isdigit():
            return int(raw)
        medline = flatten(article.findtext('.//JournalIssue/PubDate/MedlineDate', default=''))
        if medline[:4].isdigit():
            return int(medline[:4])
        return None

    @staticmethod
    def _doi(article):
        for node in article.findall('.//ArticleIdList/ArticleId'):
            if (node.get('IdType') or '').lower() == 'doi':
                text = flatten(''.join(node.itertext()))
                if text:
                    return text
        for node in article.findall('.//ELocationID'):
            if (node.get('EIdType') or '').lower() == 'doi':
                text = flatten(''.join(node.itertext()))
                if text:
                    return text
        return None
