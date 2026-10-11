import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.dedupe import dedupe
from app.sources import build_sources
from eval.metrics import mean, percentile, recall_at_k, reciprocal_rank


def main(argv=None):
    parser = argparse.ArgumentParser(description='LitScope retrieval eval (Recall@k, MRR, latency)')
    parser.add_argument('--eval-set', default='eval/eval_set.json')
    parser.add_argument('--sources', default='arxiv,crossref,pubmed')
    parser.add_argument('--top-k', default='1,5,10')
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--sleep', type=float, default=0.34)
    parser.add_argument('--max-items', type=int, default=0)
    parser.add_argument('--out', default='metrics.json')
    args = parser.parse_args(argv)

    payload = json.loads(Path(args.eval_set).read_text(encoding='utf-8'))
    items = payload.get('items') or []
    if args.max_items:
        items = items[:args.max_items]
    ks = [int(value) for value in args.top_k.split(',') if value.strip()]
    sources = build_sources([name.strip() for name in args.sources.split(',') if name.strip()])

    per_k = {k: [] for k in ks}
    rrs = []
    latencies = []
    failures = []
    details = []

    for index, item in enumerate(items, 1):
        query = item['query']
        expected = item.get('expected') or []
        started = time.perf_counter()
        pooled = []
        try:
            for source in sources:
                pooled.extend(source.search(query, args.limit))
                if args.sleep:
                    time.sleep(args.sleep)
        except Exception as exc:
            failures.append({'query': query, 'error': repr(exc)})
            print('[fail] ' + query[:60], file=sys.stderr)
            continue
        latencies.append(time.perf_counter() - started)
        ranked = [paper.uid for paper in dedupe(pooled)]
        for k in ks:
            per_k[k].append(recall_at_k(ranked, expected, k))
        rrs.append(reciprocal_rank(ranked, expected))
        rank = (ranked.index(expected[0]) + 1) if expected and expected[0] in ranked else 0
        details.append({'query': query, 'expected': expected, 'rank': rank,
                        'pooled': len(pooled), 'ranked': len(ranked)})
        print('[' + str(index) + '/' + str(len(items)) + '] rank=' + str(rank)
              + '  pooled=' + str(len(pooled)) + '  ' + query[:58], file=sys.stderr)

    scored = len(rrs)
    report = {
        'eval_set': args.eval_set,
        'label_scheme': payload.get('label_scheme'),
        'sources': [source.name for source in sources],
        'items': len(items),
        'scored': scored,
        'failures': len(failures),
        'recall': {('at_' + str(k)): round(mean(per_k[k]), 4) for k in ks},
        'mrr': round(mean(rrs), 4),
        'latency_p50_s': round(percentile(latencies, 0.5), 3) if latencies else None,
        'latency_p95_s': round(percentile(latencies, 0.95), 3) if latencies else None,
        'per_item': details,
        'errors': failures[:10],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    print('')
    print('items=' + str(report['items']) + ' scored=' + str(scored) + ' failures=' + str(report['failures']))
    print('recall: ' + json.dumps(report['recall']))
    print('mrr=' + str(report['mrr'])
          + '  p50=' + str(report['latency_p50_s']) + 's p95=' + str(report['latency_p95_s']) + 's')
    print('written to ' + str(out))
    return 0 if scored else 2


if __name__ == '__main__':
    raise SystemExit(main())
