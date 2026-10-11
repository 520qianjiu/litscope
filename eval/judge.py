import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.llm import LLMClient

JUDGE_SYSTEM = (
    'You are a strict citation auditor for scientific writing. '
    'You receive a CLAIM and a SOURCE passage. Decide whether the SOURCE supports the CLAIM. '
    'Reply with exactly one word: SUPPORTED or UNSUPPORTED. '
    'SUPPORTED requires that the source states the claim or directly entails it with concrete content. '
    'Topic-level overlap, vague similarity, or a source that merely mentions the subject is UNSUPPORTED.'
)

JUDGE_TEMPLATE = 'CLAIM:\n{claim}\n\nSOURCE:\n{source}\n\nVerdict:'
VALID_VERDICTS = ('SUPPORTED', 'UNSUPPORTED')


def parse_verdict(raw: str) -> str:
    text = (raw or '').strip().upper()
    if 'UNSUPPORTED' in text:
        return 'UNSUPPORTED'
    if 'SUPPORTED' in text:
        return 'SUPPORTED'
    return 'UNCLEAR'


def judge_pair(claim: str, source_text: str, llm: LLMClient) -> dict:
    prompt = JUDGE_TEMPLATE.format(claim=claim.strip(), source=(source_text or '')[:2000])
    started = time.perf_counter()
    response = llm.complete(prompt, system=JUDGE_SYSTEM, temperature=0.0, max_tokens=8)
    return {
        'verdict': parse_verdict(response.text),
        'raw': response.text,
        'latency': round(time.perf_counter() - started, 3),
    }


def collect_pairs(result: dict, max_claims: int = 0):
    by_number = {item['n']: item for item in result['citations']['valid']}
    pairs = []
    for claim in result['citations']['claims']:
        for number in claim['markers']:
            item = by_number.get(number)
            if item is None:
                continue
            pairs.append((claim['claim'], item))
    if max_claims:
        pairs = pairs[:max_claims]
    return pairs


def audit_answer(result: dict, llm: LLMClient, max_claims: int = 3) -> dict:
    pairs = collect_pairs(result, max_claims)
    details = []
    for claim, item in pairs:
        verdict = judge_pair(claim, item.get('text') or '', llm)
        details.append({
            'claim': claim,
            'n': item['n'],
            'paper_uid': item.get('paper_uid'),
            'verdict': verdict['verdict'],
            'raw': verdict['raw'],
        })
    supported = sum(1 for row in details if row['verdict'] == 'SUPPORTED')
    unclear = sum(1 for row in details if row['verdict'] == 'UNCLEAR')
    total = len(details)
    return {
        'pairs': total,
        'supported': supported,
        'unclear': unclear,
        'citation_support_rate': round(supported / float(total), 4) if total else 0.0,
        'detail': details,
    }
