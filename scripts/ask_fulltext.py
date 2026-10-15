"""Compare abstract-only retrieval with two-stage full-text retrieval.

    python scripts/ask_fulltext.py --topic "graph neural network" --papers 6 \
        --question "mini-batch and full-graph training differ in what way?"

The point of the comparison is that some questions simply cannot be answered from
abstracts: hyperparameters, dataset sizes and per-section claims only exist in the
body text.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.dedupe import dedupe
from app.fulltext import load_many
from app.index import build_embedder
from app.index.pipeline import build_fulltext_index, build_index
from app.index.retrievers import DenseRetriever, TwoStageRetriever
from app.llm import build_llm
from app.rag import ask_with_retriever
from app.sources import build_sources

CACHE_DIR = Path(__file__).resolve().parents[1] / '.cache' / 'fulltext'


def build_corpus(topic, per_source, sources):
    pooled = []
    for source in sources:
        try:
            found = source.search(topic, per_source)
            print('[source] %-9s -> %d' % (source.name, len(found)), file=sys.stderr)
            pooled.extend(found)
        except Exception as exc:
            print('[source] %-9s failed: %r' % (source.name, exc), file=sys.stderr)
    return dedupe(pooled)


def show(label, result):
    print('')
    print('=' * 72)
    print(label)
    print('=' * 72)
    print(result.get('answer') or '(empty answer)')
    print('')
    print('citations: %d valid / %d invalid' % (len(result['citations']['valid']),
                                                len(result['citations']['invalid'])))
    for item in result['citations']['valid'][:6]:
        section = item.get('section') or ''
        head = '[%s] %s' % (item['n'], item['paper_uid'])
        if section:
            head += '  §' + section
        print('  ' + head)
        print('      ' + str(item['text'])[:120].replace('\n', ' '))


def main(argv=None):
    parser = argparse.ArgumentParser(description='LitScope full-text vs abstract comparison')
    parser.add_argument('--topic', required=True)
    parser.add_argument('--question', required=True)
    parser.add_argument('--papers', type=int, default=6)
    parser.add_argument('--sources', default='arxiv,crossref,pubmed')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--candidates', type=int, default=4, help='stage-one shortlist size')
    parser.add_argument('--cache-dir', default=str(CACHE_DIR))
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args(argv)

    sources = build_sources([n.strip() for n in args.sources.split(',') if n.strip()])
    papers = build_corpus(args.topic, args.papers, sources)
    print('[corpus] %d papers after dedupe' % len(papers), file=sys.stderr)

    embedder = build_embedder()
    llm = build_llm()

    print('[fulltext] fetching PDFs (cached in .cache/fulltext) ...', file=sys.stderr)
    started = time.time()
    fulltexts = load_many(papers, cache_dir=args.cache_dir or None)
    usable = sum(1 for f in fulltexts.values() if f.ok)
    print('[fulltext] %d/%d papers parsed in %.1fs' % (usable, len(papers), time.time() - started),
          file=sys.stderr)
    for uid, fulltext in fulltexts.items():
        if not fulltext.ok:
            print('    skip %s (%s)' % (uid, fulltext.error), file=sys.stderr)

    abstract_store = build_index(papers, embedder)
    fulltext_store, stats = build_fulltext_index(papers, embedder, fulltexts)
    print('[index] abstract chunks=%d  fulltext chunks=%d  sections=%d'
          % (len(abstract_store), stats['chunks'], len(stats['sections'])), file=sys.stderr)

    abstract_result = ask_with_retriever(
        args.question, DenseRetriever(abstract_store, embedder, name='abstract-dense'),
        llm, top_k=args.top_k)

    two_stage = TwoStageRetriever(abstract_store, fulltext_store, embedder,
                                  candidates=args.candidates, name='two-stage-fulltext')
    fulltext_result = ask_with_retriever(args.question, two_stage, llm, top_k=args.top_k)

    if args.json:
        print(json.dumps({'question': args.question,
                          'abstract_only': abstract_result,
                          'fulltext_two_stage': fulltext_result,
                          'fulltext_stats': stats}, ensure_ascii=False, indent=2))
        return 0

    print('')
    print('QUESTION: ' + args.question)
    show('A. abstract-only retrieval', abstract_result)
    show('B. two-stage full-text retrieval (stage-one shortlist: %s)'
         % ', '.join(two_stage.last_shortlist), fulltext_result)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
