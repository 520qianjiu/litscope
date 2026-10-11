import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.dedupe import dedupe
from app.index import build_embedder
from app.index.bm25 import tokenize
from app.index.pipeline import build_index
from app.index.retrievers import BM25Retriever, DenseRetriever
from app.llm import build_llm
from app.sources import build_sources
from eval.metrics import mean, percentile, recall_at_k, reciprocal_rank

STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'of', 'to', 'in', 'on', 'for', 'with', 'by',
    'is', 'are', 'was', 'were', 'be', 'been', 'as', 'at', 'from', 'that', 'this', 'these',
    'those', 'it', 'its', 'we', 'our', 'they', 'their', 'has', 'have', 'had', 'can', 'may',
    'which', 'while', 'also', 'such', 'than', 'then', 'into', 'using', 'used', 'use', 'more',
    'most', 'not', 'no', 'however', 'thus', 'between', 'within', 'both', 'each', 'all',
}

CACHE_PATH = Path(__file__).resolve().parent / 'zh_titles.json'
TRANSLATE_SYSTEM = ('You translate scientific paper titles into Simplified Chinese. '
                    'Reply with the translation only. No quotes, no explanation, no pinyin.')


def keyword_query(abstract: str, limit: int = 10) -> str:
    words = [w for w in tokenize(abstract or '') if w not in STOPWORDS and len(w) > 2]
    return ' '.join(words[:limit])


def load_cache() -> dict:
    if CACHE_PATH.is_file():
        return json.loads(CACHE_PATH.read_text(encoding='utf-8'))
    return {}


def save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding='utf-8')


def zh_query(llm, title: str, cache: dict) -> str:
    if title in cache:
        return cache[title]
    response = llm.complete('Title: ' + title, system=TRANSLATE_SYSTEM,
                            temperature=0.0, max_tokens=120)
    text = ' '.join((response.text or '').split())
    cache[title] = text
    return text


def build_items(papers, mode, limit, llm=None, cache=None):
    items = []
    for paper in papers[:limit]:
        if mode == 'title':
            query = paper.title
        elif mode == 'zh':
            if llm is None:
                raise ValueError('zh query mode needs an LLM client')
            query = zh_query(llm, paper.title, cache if cache is not None else {})
        else:
            query = keyword_query(paper.abstract or paper.title)
        if not (query or '').strip():
            continue
        items.append({'query': query, 'expected': [paper.uid], 'title': paper.title})
    return items


def evaluate(retriever, items, top_k: int) -> dict:
    recalls = []
    rrs = []
    top_scores = []
    latencies = []
    empty_hits = 0
    rows = []
    for item in items:
        started = time.perf_counter()
        hits = retriever.search(item['query'], top_k)
        latencies.append(time.perf_counter() - started)
        if not hits:
            empty_hits += 1
        uids = [record.get('paper_uid') for _score, record in hits]
        recall = recall_at_k(uids, item['expected'], top_k)
        rr = reciprocal_rank(uids, item['expected'])
        recalls.append(recall)
        rrs.append(rr)
        top_scores.append(float(hits[0][0]) if hits else 0.0)
        rows.append({'query': item['query'], 'expected': item['expected'],
                     'ranked': uids, 'recall': recall, 'rr': round(rr, 4)})
    return {
        'retriever': retriever.name,
        'items': len(items),
        'empty_results': empty_hits,
        'recall_at_k': round(mean(recalls), 4),
        'mrr': round(mean(rrs), 4),
        'mean_top_score': round(mean(top_scores), 4),
        'latency_p50_ms': round(percentile(latencies, 0.5) * 1000, 2),
        'latency_p95_ms': round(percentile(latencies, 0.95) * 1000, 2),
        'per_item': rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description='Compare retrieval backends on the same corpus')
    parser.add_argument('--topic', default='graph neural network')
    parser.add_argument('--papers', type=int, default=8)
    parser.add_argument('--sources', default='arxiv,crossref,pubmed')
    parser.add_argument('--retrievers', default='bm25,dense')
    parser.add_argument('--query-mode', default='keywords', choices=['keywords', 'title', 'zh'])
    parser.add_argument('--questions', type=int, default=12)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--label', default='')
    parser.add_argument('--out', default='metrics_compare.json')
    args = parser.parse_args(argv)

    sources = build_sources([name.strip() for name in args.sources.split(',') if name.strip()])
    pooled = []
    for source in sources:
        try:
            pooled.extend(source.search(args.topic, args.papers))
        except Exception as exc:
            print('[source fail] ' + source.name + ' ' + repr(exc), file=sys.stderr)
    corpus = dedupe(pooled)
    embedder = build_embedder()
    store = build_index(corpus, embedder)

    llm = None
    cache = {}
    if args.query_mode == 'zh':
        llm = build_llm()
        cache = load_cache()
    items = build_items(corpus, args.query_mode, args.questions, llm=llm, cache=cache)
    if args.query_mode == 'zh':
        save_cache(cache)

    print('corpus=' + str(len(corpus)) + ' papers, index=' + str(len(store)) + ' chunks, '
          + 'embedder=' + embedder.name + ', queries=' + str(len(items))
          + ' (' + args.query_mode + ')', file=sys.stderr)

    results = []
    for kind in [name.strip() for name in args.retrievers.split(',') if name.strip()]:
        if kind.lower() == 'bm25':
            retriever = BM25Retriever.from_store(store)
        else:
            retriever = DenseRetriever(store, embedder)
        report = evaluate(retriever, items, args.top_k)
        results.append(report)
        print('  ' + report['retriever'] + ': recall@' + str(args.top_k) + '='
              + str(report['recall_at_k']) + ' mrr=' + str(report['mrr'])
              + ' empty=' + str(report['empty_results']), file=sys.stderr)

    payload = {
        'label': args.label,
        'topic': args.topic,
        'query_mode': args.query_mode,
        'corpus_papers': len(corpus),
        'index_chunks': len(store),
        'embedder': embedder.name,
        'top_k': args.top_k,
        'results': results,
    }
    out = Path(args.out)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    print('')
    header = ('retriever'.ljust(26) + 'recall@' + str(args.top_k).ljust(4)
              + 'mrr'.ljust(10) + 'empty'.ljust(8) + 'mean_top_score')
    print(header)
    print('-' * len(header))
    for row in results:
        print(row['retriever'].ljust(26)
              + str(row['recall_at_k']).ljust(10)
              + str(row['mrr']).ljust(10)
              + str(row['empty_results']).ljust(8)
              + str(row['mean_top_score']))
    print('')
    print('sample queries:')
    for item in items[:3]:
        print('   ' + item['query'][:90])
    print('written to ' + str(out))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
