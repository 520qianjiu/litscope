from typing import Dict, List, Sequence

SYSTEM_PROMPT = (
    'You are LitScope, a literature assistant for researchers. '
    'Answer ONLY from the numbered sources provided by the user. '
    'Every factual sentence must end with citation markers such as [1] or [2][3]. '
    'If the sources do not support an answer, say so explicitly instead of guessing. '
    'Never invent a reference or a citation number that is not in the source list. '
    'Reply in Chinese when the question is in Chinese, otherwise reply in English.'
)


def format_source(number: int, record: Dict) -> str:
    label = str(record.get('paper_uid') or 'unknown')
    text = str(record.get('text') or '')
    return '[' + str(number) + '] ' + label + '\n' + text


def build_context_block(contexts: Sequence[Dict]) -> str:
    parts = [format_source(item['n'], item['record']) for item in contexts]
    return '\n\n'.join(parts)


def build_messages(question: str, contexts: Sequence[Dict]) -> List[Dict]:
    user = ('Sources:\n\n' + build_context_block(contexts)
            + '\n\nQuestion: ' + question
            + '\n\nAnswer with inline citations.')
    return [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user},
    ]
