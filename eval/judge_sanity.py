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
from eval.judge import collect_pairs, judge_pair
from eval.metrics import mean

QUESTION_TEMPLATE = '总结论文「{title}」的主要贡献与方法。'


def main(argv=None):
    parser = argparse.ArgumentParser(description='Negative control: can the judge reject unsupported citations?')
    parser.add_argument('--topic', default='graph neural network')
    parser.add_argument('--papers', type=int, default=8)
    parser.add_argument('--questions', type=int, default=6)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--out', default='metrics_judge_sanity.json')
    args = parser.parse_args(argv)

    sources = build_sources(['arxiv', 'crossref', 'pubmed'])
    pooled = []
    for source in sources:
        try:
            pooled.extend(source.search(args.topic, args.papers))
        except Exception as exc:
            print('[source fail] ' + source.name, file=sys.stderr)

    corpus = dedupe(pooled)
    embedder = build_embedder()
    store = build_index(corpus, embedder)
    llm = build_llm()
    print('corpus=' + str(len(corpus)) + ' index=' + str(len(store)), file=sys.stderr)

    samples = []
    for paper in corpus[:args.questions]:
        question = QUESTION_TEMPLATE.format(title=paper.title)
        result = ask(question, store, embedder, llm, top_k=args.top_k)
        pairs = collect_pairs(result, 1)
        if not pairs:
            continue
        claim, item = pairs[0]
        samples.append({
            'claim': claim,
            'own_text': item.get('text') or '',
            'own_uid': item.get('paper_uid'),
        })

    if len(samples) < 2:
        print('not enough samples to build a control')
        return 2

    rows = []
    for index, sample in enumerate(samples):
        foreign = samples[(index + 1) % len(samples)]
        if foreign['own_uid'] == sample['own_uid']:
            foreign = samples[(index + 2) % len(samples)]
        positive = judge_pair(sample['claim'], sample['own_text'], llm)
        negative = judge_pair(sample['claim'], foreign['own_text'], llm)
        rows.append({
            'claim': sample['claim'],
            'own_uid': sample['own_uid'],
            'foreign_uid': foreign['own_uid'],
            'own_verdict': positive['verdict'],
            'foreign_verdict': negative['verdict'],
        })
        print('[' + str(index + 1) + '] own=' + positive['verdict']
              + '  foreign=' + negative['verdict'], file=sys.stderr)

    # balanced accuracy: a rubber-stamp judge (always SUPPORTED, never rejects) scores 0.5
    positive_rate = mean([1.0 if row['own_verdict'] == 'SUPPORTED' else 0.0 for row in rows])
    negative_rejection = mean([1.0 if row['foreign_verdict'] == 'UNSUPPORTED' else 0.0 for row in rows])
    report = {
        'samples': len(rows),
        'positive_support_rate': round(positive_rate, 4),
        'negative_rejection_rate': round(negative_rejection, 4),
        'judge_balanced_accuracy': round((positive_rate + negative_rejection) / 2.0, 4),
        'judge_bias': round(positive_rate - negative_rejection, 4),
        'note': 'positive = claim paired with its own source; negative = same claim paired with a different paper source',
        'rows': rows,
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('')
    print('samples                  = ' + str(report['samples']))
    print('positive_support_rate    = ' + str(report['positive_support_rate']))
    print('negative_rejection_rate  = ' + str(report['negative_rejection_rate']))
    print('judge_balanced_accuracy  = ' + str(report['judge_balanced_accuracy']))
    print('    (0.5 = rubber stamp, 1.0 = perfect discrimination)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
