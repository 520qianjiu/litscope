import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.dedupe import dedupe
from app.index import build_embedder
from app.index.pipeline import build_index
from app.llm import build_llm
from app.rag import ask
from app.sources import build_sources


def build_corpus(topic, limit_per_source, sources):
    pooled = []
    for source in sources:
        try:
            found = source.search(topic, limit_per_source)
            print('[source] ' + source.name + ' -> ' + str(len(found)), file=sys.stderr)
            pooled.extend(found)
        except Exception as exc:
            print('[source] ' + source.name + ' failed: ' + repr(exc), file=sys.stderr)
    return dedupe(pooled)


def main(argv=None):
    parser = argparse.ArgumentParser(description='LitScope: ask a question against a freshly built corpus')
    parser.add_argument('--topic', required=True, help='corpus seed topic')
    parser.add_argument('--question', required=True)
    parser.add_argument('--papers', type=int, default=8, help='per-source paper limit')
    parser.add_argument('--sources', default='arxiv,crossref,pubmed')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args(argv)

    sources = build_sources([name.strip() for name in args.sources.split(',') if name.strip()])
    papers = build_corpus(args.topic, args.papers, sources)
    print('[corpus] ' + str(len(papers)) + ' papers after dedupe', file=sys.stderr)

    embedder = build_embedder()
    store = build_index(papers, embedder)
    print('[index] ' + str(len(store)) + ' chunks with ' + embedder.name, file=sys.stderr)

    llm = build_llm()
    result = ask(args.question, store, embedder, llm, top_k=args.top_k)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print('')
    print('QUESTION: ' + args.question)
    print('')
    print(result['answer'] or '(空回答)')
    print('')
    print('--- citations ---')
    for item in result['citations']['valid']:
        print('[' + str(item['n']) + '] ' + str(item['paper_uid'])
              + '  chunk#' + str(item['chunk_index']) + '  score=' + str(item['score']))
        print('    ' + str(item['text'])[:160])
    if result['citations']['invalid']:
        print('INVALID markers: ' + str(result['citations']['invalid']))
    usage = result.get('usage') or {}
    print('')
    print('--- usage ---')
    print('model=' + str(usage.get('model')) + '  tokens=' + str(usage.get('total_tokens'))
          + '  latency=' + str(usage.get('latency')) + 's')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
