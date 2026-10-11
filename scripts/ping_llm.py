import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config


def main():
    if not config.DEEPSEEK_API_KEY:
        print('DEEPSEEK_API_KEY 未设置，请先填 .env')
        return 2
    payload = {
        'model': config.DEEPSEEK_MODEL,
        'messages': [{'role': 'user', 'content': '只回复两个字：通了'}],
        'max_tokens': 16,
        'stream': False,
    }
    request = urllib.request.Request(
        config.DEEPSEEK_BASE_URL.rstrip('/') + '/chat/completions',
        data=json.dumps(payload).encode('utf-8'),
        headers={
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + config.DEEPSEEK_API_KEY,
        },
        method='POST',
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode('utf-8', errors='replace'))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode('utf-8', errors='replace')[:400]
        print('HTTP ' + str(exc.code) + ' -> ' + body)
        if exc.code == 401:
            print('=> key 无效或已被删除')
        elif exc.code == 402:
            print('=> 余额不足，需要充值')
        elif exc.code == 429:
            print('=> 触发限流')
        return 1
    except Exception as exc:
        print('请求失败: ' + repr(exc))
        return 1
    elapsed = time.perf_counter() - started
    choice = (data.get('choices') or [{}])[0]
    content = ((choice.get('message') or {}).get('content') or '').strip()
    usage = data.get('usage') or {}
    print('reply   : ' + content)
    print('model   : ' + str(data.get('model')))
    print('latency : ' + str(round(elapsed, 2)) + 's')
    print('tokens  : prompt=' + str(usage.get('prompt_tokens'))
          + ' completion=' + str(usage.get('completion_tokens'))
          + ' total=' + str(usage.get('total_tokens')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
