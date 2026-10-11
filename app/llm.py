import json
import time
import urllib.error
import urllib.request
from typing import Dict, List, Sequence

from . import config

RETRYABLE_STATUS = (429, 500, 502, 503, 504)


class LLMError(RuntimeError):
    pass


class LLMResponse:
    def __init__(self, text: str, model: str = '', prompt_tokens: int = 0,
                 completion_tokens: int = 0, latency: float = 0.0):
        self.text = text
        self.model = model
        self.prompt_tokens = int(prompt_tokens)
        self.completion_tokens = int(completion_tokens)
        self.latency = float(latency)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> Dict:
        return {
            'text': self.text,
            'model': self.model,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'total_tokens': self.total_tokens,
            'latency': round(self.latency, 3),
        }


class LLMClient:
    name = 'base'

    def chat(self, messages: Sequence[Dict], temperature: float = 0.2,
             max_tokens: int = 1024) -> LLMResponse:
        raise NotImplementedError

    def complete(self, prompt: str, system: str = '', temperature: float = 0.2,
                 max_tokens: int = 1024) -> LLMResponse:
        messages: List[Dict] = []
        if system:
            messages.append({'role': 'system', 'content': system})
        messages.append({'role': 'user', 'content': prompt})
        return self.chat(messages, temperature=temperature, max_tokens=max_tokens)


class OpenAICompatClient(LLMClient):
    def __init__(self, base_url: str, api_key: str, model: str,
                 name: str = 'openai-compatible', retries: int = 3,
                 timeout: float = 90.0):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.name = name
        self.retries = int(retries)
        self.timeout = float(timeout)

    def chat(self, messages: Sequence[Dict], temperature: float = 0.2,
             max_tokens: int = 1024) -> LLMResponse:
        payload = {
            'model': self.model,
            'messages': [dict(message) for message in messages],
            'temperature': float(temperature),
            'max_tokens': int(max_tokens),
            'stream': False,
        }
        body = json.dumps(payload).encode('utf-8')
        last_error = 'unknown'
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(
                self.base_url + '/chat/completions',
                data=body,
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer ' + self.api_key,
                },
                method='POST',
            )
            started = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode('utf-8', errors='replace'))
                latency = time.perf_counter() - started
                usage = data.get('usage') or {}
                choices = data.get('choices') or [{}]
                content = (choices[0].get('message') or {}).get('content') or ''
                return LLMResponse(
                    text=content.strip(),
                    model=str(data.get('model') or self.model),
                    prompt_tokens=usage.get('prompt_tokens') or 0,
                    completion_tokens=usage.get('completion_tokens') or 0,
                    latency=latency,
                )
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode('utf-8', errors='replace')[:300]
                last_error = 'HTTP ' + str(exc.code) + ': ' + detail
                if exc.code not in RETRYABLE_STATUS:
                    raise LLMError(last_error) from exc
            except Exception as exc:
                last_error = repr(exc)
            if attempt < self.retries:
                time.sleep(1.5 * (attempt + 1))
        raise LLMError('llm request failed after retries: ' + str(last_error))


class ScriptedClient(LLMClient):
    def __init__(self, replies=None, name: str = 'scripted'):
        self.name = name
        self.replies = list(replies or [])
        self.calls: List[List[Dict]] = []

    def chat(self, messages: Sequence[Dict], temperature: float = 0.2,
             max_tokens: int = 1024) -> LLMResponse:
        self.calls.append([dict(message) for message in messages])
        text = self.replies.pop(0) if self.replies else ''
        return LLMResponse(text=text, model=self.name)


def build_llm(name: str = '') -> LLMClient:
    provider = (name or config.LLM_PROVIDER or 'deepseek').strip().lower()
    if provider == 'deepseek':
        if not config.DEEPSEEK_API_KEY:
            raise LLMError('DEEPSEEK_API_KEY 未设置，请先填 .env')
        return OpenAICompatClient(config.DEEPSEEK_BASE_URL, config.DEEPSEEK_API_KEY,
                                  config.DEEPSEEK_MODEL, name='deepseek')
    if provider in ('openai', 'openai-compatible', 'siliconflow', 'zhipu'):
        base_url = config._value('LLM_BASE_URL')
        api_key = config._value('LLM_API_KEY')
        model = config._value('LLM_MODEL')
        if not (base_url and api_key and model):
            raise LLMError('LLM_BASE_URL / LLM_API_KEY / LLM_MODEL are required')
        return OpenAICompatClient(base_url, api_key, model, name=provider)
    raise ValueError('unknown llm provider: ' + provider)
