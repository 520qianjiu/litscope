import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUOTE_CHARS = (chr(39), chr(34))


def _parse_env_file(path: Path) -> dict:
    values = {}
    try:
        text = path.read_text(encoding='utf-8')
    except OSError:
        return values
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('export '):
            line = line[7:].strip()
        if '=' not in line:
            continue
        key, _, value = line.partition('=')
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in QUOTE_CHARS:
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def load_env(path=None) -> int:
    env_path = Path(path) if path else Path(os.environ.get('LITSCOPE_ENV') or (PROJECT_ROOT / '.env'))
    loaded = {}
    try:
        from dotenv import dotenv_values
        loaded = {k: v for k, v in (dotenv_values(env_path) or {}).items() if v is not None}
    except Exception:
        loaded = _parse_env_file(env_path)
    count = 0
    for key, value in loaded.items():
        if key and key not in os.environ:
            os.environ[key] = str(value)
            count += 1
    return count


LOADED_FROM_DOTENV = load_env()


def _value(name: str, default: str = '') -> str:
    return (os.environ.get(name) or default).strip()


def _as_float(name: str, default: float) -> float:
    raw = _value(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


HTTP_TIMEOUT = _as_float('HTTP_TIMEOUT', 30.0)
USER_AGENT = _value('LITSCOPE_USER_AGENT',
                    'LitScope/0.1 (academic literature assistant; contact: you@example.com)')

PUBMED_EMAIL = _value('PUBMED_EMAIL')
PUBMED_API_KEY = _value('PUBMED_API_KEY')

LLM_PROVIDER = _value('LLM_PROVIDER', 'deepseek').lower()
DEEPSEEK_API_KEY = _value('DEEPSEEK_API_KEY')
DEEPSEEK_BASE_URL = _value('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
DEEPSEEK_MODEL = _value('DEEPSEEK_MODEL', 'deepseek-chat')

EMBEDDING_PROVIDER = _value('EMBEDDING_PROVIDER', 'hashing')

SOURCE_NAMES = ['arxiv', 'crossref', 'pubmed']
