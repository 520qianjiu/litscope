import re
from typing import Dict, List, Sequence

MARKER = re.compile(r'\[(\d+)\]')
SENTENCE_END = re.compile(r'(?<=[.!?。！？])')


def extract_markers(text: str) -> List[int]:
    numbers: List[int] = []
    for match in MARKER.finditer(text or ''):
        value = int(match.group(1))
        if value not in numbers:
            numbers.append(value)
    return numbers


def split_claims(text: str) -> List[str]:
    pieces = [piece.strip() for piece in SENTENCE_END.split(text or '')]
    return [piece for piece in pieces if piece and MARKER.search(piece)]


def resolve_citations(text: str, contexts: Sequence[Dict]) -> Dict:
    by_number = {int(item['n']): item for item in contexts}
    resolved = []
    invalid = []
    for number in extract_markers(text):
        item = by_number.get(number)
        if item is None:
            invalid.append(number)
            continue
        record = item.get('record') or {}
        resolved.append({
            'n': number,
            'paper_uid': record.get('paper_uid'),
            'chunk_uid': record.get('uid'),
            'chunk_index': record.get('chunk_index'),
            'score': item.get('score'),
            'text': record.get('text'),
        })
    claims = []
    for claim in split_claims(text):
        claims.append({'claim': claim, 'markers': extract_markers(claim)})
    return {
        'used': extract_markers(text),
        'valid': resolved,
        'invalid': invalid,
        'claims': claims,
    }
