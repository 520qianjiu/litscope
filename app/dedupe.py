import re
from difflib import SequenceMatcher
from typing import Iterable, List

from .models import Paper

_NON_ALNUM = re.compile(r'[^0-9a-z\u4e00-\u9fff]+')
FUZZY_THRESHOLD = 0.92


def norm_title(title: str) -> str:
    return _NON_ALNUM.sub('', (title or '').lower())


def similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left, right).ratio()


def merge_into(keep: Paper, extra: Paper) -> None:
    if not keep.abstract and extra.abstract:
        keep.abstract = extra.abstract
    if not keep.doi and extra.doi:
        keep.doi = extra.doi
    if not keep.year and extra.year:
        keep.year = extra.year
    if not keep.venue and extra.venue:
        keep.venue = extra.venue
    if not keep.url and extra.url:
        keep.url = extra.url
    if keep.citations is None or (extra.citations or 0) > (keep.citations or 0):
        keep.citations = extra.citations
    seen = set(keep.authors)
    for name in extra.authors:
        if name not in seen:
            keep.authors.append(name)
            seen.add(name)
    tag = extra.source + ':' + extra.source_id
    if tag not in keep.merged_sources and tag != keep.source + ':' + keep.source_id:
        keep.merged_sources.append(tag)


def dedupe(papers: Iterable[Paper], fuzzy_threshold: float = FUZZY_THRESHOLD) -> List[Paper]:
    kept: List[Paper] = []
    by_doi = {}
    by_title = {}
    for paper in papers:
        title = (paper.title or '').strip()
        if not title:
            continue
        doi = (paper.doi or '').strip().lower() or None
        norm = norm_title(title)
        hit = None
        if doi and doi in by_doi:
            hit = by_doi[doi]
        elif norm and norm in by_title:
            hit = by_title[norm]
        else:
            for candidate in kept:
                other = norm_title(candidate.title)
                if other and norm and similarity(other, norm) >= fuzzy_threshold:
                    hit = candidate
                    break
        if hit is None:
            hit = paper
            kept.append(hit)
        else:
            merge_into(hit, paper)
        if doi:
            by_doi.setdefault(doi, hit)
        key = norm or norm_title(hit.title)
        if key:
            by_title.setdefault(key, hit)
    return kept
