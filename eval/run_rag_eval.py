import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.dedupe import dedupe
from app.index import build_embedder
from app.index.pipeline import build_index
from app.llm import build_llm
from app.rag import ask
from app.sources import build_sources
from eval.judge import audit_answer
from eval.metrics import mean, percentile

QUESTION_TEMPLATE = '总结论文「{title}」的主要贡献与方法。'


def main(argv=None):
    parser = argparse.ArgumentParser(description='LitScope RAG end-to-end eval with LLM-as-judge')
    parser.add_argument('--topic', default='language model')
    parser.add_argument('--papers', type=int, default=8)
    parser.add_argument('--questions', type=int, default=6)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--max-claims', type=int, default=3)
    parser.add_argument('--sources', default='arxiv,crossref,pubmed')
    parser.add_argument('--out', default='metrics_rag.json')
    args = parser.parse_args(argv)

    sources = build_sources([name.strip() for name in args.sources.split(',') if name.strip()])
    pooled = []
    for source in sources:
        try:
            pooled.extend(source.search(args.topic, args.papers))
        except Exception as exc:
            print('[source fail] ' + source.name + ' ' + repr(exc), file=sys.stderr)
    corpus = dedupe(pooled)
    print('corpus=' + str(len(corpus)) + ' papers', file=sys.stderr)

    embedder = build_embedder()
    store = build_index(corpus, embedder)
    print('index=' + str(len(store)) + ' chunks (' + embedder.name + ')', file=sys.stderr)

    llm = build_llm()
    sample = corpus[:args.questions]

    latencies = []
    tokens = []
    support_rates = []
    invalid_markers = 0
    target_hits = 0
    rows = []
    judged_pairs = 0

    for index, paper in enumerate(sample, 1):
        question = QUESTION_TEMPLATE.format(title=paper.title)
        started = time.perf_counter()
        result = ask(question, store, embedder, llm, top_k=args.top_k)
        elapsed = time.perf_counter() - started
        latencies.append(elapsed)

        usage = result.get('usage') or {}
        tokens.append(usage.get('total_tokens') or 0)
        invalid_markers += len(result['citations']['invalid'])

        retrieved = [item['record'].get('paper_uid') for item in result['contexts']]
        hit = paper.uid in retrieved
        target_hits += 1 if hit else 0

        audit = audit_answer(result, llm, max_claims=args.max_claims)
        judged_pairs += audit['pairs']
        if audit['pairs']:
            support_rates.append(audit['citation_support_rate'])

        rows.append({
            'question': question,
            'target_uid': paper.uid,
            'target_retrieved': hit,
            'answer_chars': len(result['answer']),
            'citations_used': result['citations']['used'],
            'citations_invalid': result['citations']['invalid'],
            'audit': audit,
            'latency_s': round(elapsed, 3),
            'tokens': usage.get('total_tokens'),
        })
        print('[' + str(index) + '/' + str(len(sample)) + '] hit=' + str(hit)
              + ' support=' + str(audit['citation_support_rate'])
              + ' pairs=' + str(audit['pairs'])
              + '  ' + question[:44], file=sys.stderr)

    total = len(sample)
    report = {
        'topic': args.topic,
        'corpus_papers': len(corpus),
        'index_chunks': len(store),
        'embedder': embedder.name,
        'llm': getattr(llm, 'name', 'unknown'),
        'questions': total,
        'top_k': args.top_k,
        'target_retrieval_rate': round(target_hits / float(total), 4) if total else 0.0,
        'citation_support_rate_mean': round(mean(support_rates), 4),
        'judged_pairs': judged_pairs,
        'invalid_marker_count': invalid_markers,
        'latency_p50_s': round(percentile(latencies, 0.5), 3) if latencies else None,
        'latency_p95_s': round(percentile(latencies, 0.95), 3) if latencies else None,
        'tokens_total': sum(tokens),
        'tokens_mean': round(mean(tokens), 1),
        'per_question': rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    print('')
    print('target_retrieval_rate   = ' + str(report['target_retrieval_rate']))
    print('citation_support_rate   = ' + str(report['citation_support_rate_mean'])
          + '  (judged pairs=' + str(judged_pairs) + ')')
    print('invalid_markers         = ' + str(invalid_markers))
    print('latency p50=' + str(report['latency_p50_s']) + 's  p95=' + str(report['latency_p95_s']) + 's')
    print('tokens total=' + str(report['tokens_total']) + ' mean=' + str(report['tokens_mean']))
    print('written to ' + str(out))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
