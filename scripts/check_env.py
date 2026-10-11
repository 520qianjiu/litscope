import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config


def mask(value):
    if not value:
        return '(未设置)'
    if len(value) <= 8:
        return value[:2] + '***'
    return value[:4] + '...' + value[-4:] + ' (len=' + str(len(value)) + ')'


print('env file: ' + str(config.PROJECT_ROOT / '.env'))
print('loaded keys from .env: ' + str(config.LOADED_FROM_DOTENV))
print('')
print('DEEPSEEK_API_KEY  = ' + mask(config.DEEPSEEK_API_KEY))
print('DEEPSEEK_BASE_URL = ' + config.DEEPSEEK_BASE_URL)
print('DEEPSEEK_MODEL    = ' + config.DEEPSEEK_MODEL)
print('EMBEDDING_PROVIDER= ' + config.EMBEDDING_PROVIDER)
print('PUBMED_EMAIL      = ' + mask(config.PUBMED_EMAIL))
print('')
print('LLM ready: ' + ('YES' if config.DEEPSEEK_API_KEY else 'NO - 请先把 key 填进 .env'))
