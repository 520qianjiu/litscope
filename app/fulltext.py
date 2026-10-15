"""Full-text acquisition and section-aware parsing.

Only arXiv is supported: its PDFs are freely and legally downloadable, and every
other source keeps the abstract-only path. The module deliberately imports PyMuPDF
lazily so that importing it (and unit-testing the section logic) never needs the
dependency installed.
"""

import os
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .config import HTTP_TIMEOUT

ARXIV_PDF_BASE = 'https://arxiv.org/pdf/'
USER_AGENT = 'LitScope/0.1 (research prototype; mailto:litscope@example.com)'

# A heading must contain one of these as a whole word. The list is deliberately
# curated: it is what separates real section titles from table cells such as
# GPU / ACC / DGL, which also extract as short ALL-CAPS lines.
SECTION_KEYWORDS = (
    'abstract', 'introduction', 'background', 'related', 'preliminar',
    'methodolog', 'method', 'approach', 'design', 'model', 'implementation',
    'setup', 'experiment', 'evaluation', 'results', 'analysis', 'discussion',
    'conclusion', 'limitation', 'future', 'reference', 'appendix',
    'acknowledg', 'dataset', 'system',
)

MIN_HEADING_CHARS = 4
MAX_HEADING_CHARS = 60
MAX_HEADING_WORDS = 8


@dataclass
class Section:
    name: str
    text: str

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass
class FullText:
    source_id: str
    sections: List[Section] = field(default_factory=list)
    raw_chars: int = 0
    error: str = ''

    @property
    def ok(self) -> bool:
        return bool(self.sections)

    def char_count(self) -> int:
        return sum(s.char_count for s in self.sections)


def normalize(line: str) -> str:
    return ' '.join((line or '').split())


def is_heading(line: str) -> bool:
    """True for ACM/IEEE-style unnumbered ALL-CAPS section titles."""
    text = normalize(line)
    if not (MIN_HEADING_CHARS <= len(text) <= MAX_HEADING_CHARS):
        return False
    words = text.split()
    if len(words) > MAX_HEADING_WORDS:
        return False
    if any(ch.isdigit() for ch in text):
        return False
    if not any(ch.isalpha() for ch in text):
        return False
    if text.upper() != text:
        return False
    lowered = text.lower()
    return any(keyword in lowered for keyword in SECTION_KEYWORDS)


def split_sections(text: str) -> List[Section]:
    """Split extracted PDF text into sections. Text before the first heading is
    dropped only when it looks like front matter (title/authors); otherwise it is
    kept under a synthetic name so nothing silently disappears."""
    lines = (text or '').splitlines()
    marks = [i for i, line in enumerate(lines) if is_heading(line)]
    if not marks:
        body = '\n'.join(lines).strip()
        return [Section(name='FULLTEXT', text=body)] if body else []
    sections: List[Section] = []
    head = '\n'.join(lines[:marks[0]]).strip()
    if head:
        sections.append(Section(name='FRONT MATTER', text=head))
    for position, start in enumerate(marks):
        end = marks[position + 1] if position + 1 < len(marks) else len(lines)
        # The heading line stays in the body so each chunk is self-describing and
        # nothing is lost between sections.
        body = '\n'.join(lines[start:end]).strip()
        sections.append(Section(name=normalize(lines[start]), text=body))
    return [s for s in sections if s.text]


def arxiv_id(paper) -> Optional[str]:
    """Recover the bare arXiv identifier from a Paper, if it is an arXiv paper."""
    if getattr(paper, 'source', '') != 'arxiv':
        return None
    raw = (getattr(paper, 'url', '') or '') + ' ' + (getattr(paper, 'source_id', '') or '')
    match = re.search(r'(\d{4}\.\d{4,5})(v\d+)?', raw)
    if match:
        return match.group(1) + (match.group(2) or '')
    legacy = re.search(r'([a-z\-]+/\d{7})(v\d+)?', raw.lower())
    return (legacy.group(1) + (legacy.group(2) or '')) if legacy else None


def pdf_url(paper) -> Optional[str]:
    identifier = arxiv_id(paper)
    return ARXIV_PDF_BASE + identifier if identifier else None


def fetch_pdf(url: str, cache_dir: Optional[str] = None, timeout: float = None) -> bytes:
    """Download a PDF, caching to disk when a cache directory is given."""
    cache_path = None
    if cache_dir:
        cache_path = Path(cache_dir) / (re.sub(r'[^A-Za-z0-9._-]', '_', url.rsplit('/', 1)[-1]) + '.pdf')
        if cache_path.is_file() and cache_path.stat().st_size > 0:
            return cache_path.read_bytes()
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout or HTTP_TIMEOUT) as response:
        data = response.read()
    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(data)
        except OSError:
            # A read-only or missing cache directory must never break the download;
            # caching is an optimisation, not a requirement.
            pass
    return data


def pdf_to_text(data: bytes) -> str:
    import pymupdf  # lazy: keeps this module importable without the dependency

    document = pymupdf.open(stream=data, filetype='pdf')
    try:
        return '\n'.join(page.get_text() for page in document)
    finally:
        document.close()


def load_fulltext(paper, cache_dir: Optional[str] = None) -> FullText:
    """Fetch + parse one paper. Never raises: failures come back as FullText(error=...)."""
    identifier = arxiv_id(paper) or getattr(paper, 'source_id', '')
    url = pdf_url(paper)
    if not url:
        return FullText(source_id=identifier, error='not an arXiv paper')
    try:
        data = fetch_pdf(url, cache_dir=cache_dir)
    except Exception as exc:
        return FullText(source_id=identifier, error='download failed: ' + type(exc).__name__)
    try:
        text = pdf_to_text(data)
    except Exception as exc:
        return FullText(source_id=identifier, error='parse failed: ' + type(exc).__name__)
    sections = split_sections(text)
    if not sections:
        return FullText(source_id=identifier, raw_chars=len(text), error='no text extracted')
    return FullText(source_id=identifier, sections=sections, raw_chars=len(text))


def load_many(papers, cache_dir: Optional[str] = None, limit: Optional[int] = None) -> dict:
    """Fetch full texts for a batch; returns {paper_uid: FullText}."""
    out = {}
    for paper in (papers[:limit] if limit else papers):
        result = load_fulltext(paper, cache_dir=cache_dir)
        out[paper.uid] = result
    return out
