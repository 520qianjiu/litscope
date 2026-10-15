"""Quantify what full text buys over abstracts.

Labels are objective and automatic: an evidence sentence is sampled from the body of
a paper (METHODS / RESULTS). By construction it cannot appear in the abstract, so
abstract-only retrieval has a near-zero ceiling while full-text retrieval can be
measured against a real target -- no human annotation and no LLM judge for recall.
"""

import argparse
import json
import random
import re
import sys
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

# Matched as substrings rather than exact names: extracted section titles vary a
# lot between publishers ('METHODS', 'EXPERIMENTAL METHODOLOGY', '4.1 Evaluation').
EVIDENCE_HINTS = ('METHOD', 'RESULT', 'EXPERIMENT', 'EVALUATION', 'ANALYSIS', 'STUDY')
EXCLUDED_SECTIONS = ('ABSTRACT', 'TITLE', 'REFERENCES', 'FRONT MATTER', 'APPENDIX',
                     'ACKNOWLEDG', 'FIGURE', 'TABLE', 'SUPPLEMENT')
SENTENCE = re.compile(r'(?<=[.!?])\s+')
MIN_EVIDENCE = 70
MAX_EVIDENCE = 320

QUESTION_SYSTEM = ('You write one factual question that the given sentence answers. '
                   'Reply with the question only, no preamble, no quotes.')
ABSTENTION_SYSTEM = ('You decide whether an answer states that the provided sources cannot '
                     'answer the question. Reply with exactly REFUSED or ANSWERED.')


def normalize(text):
    return ' '.join((text or '').lower().split())


def usable_sections(fulltext, strict=True):
    picked = []
    for section in fulltext.sections:
        name = (section.name or '').upper()
        if strict:
            if any(hint in name for hint in EVIDENCE_HINTS):
                picked.append(section)
        elif not any(bad in name for bad in EXCLUDED_SECTIONS):
            picked.append(section)
    return picked


def pick_evidence(fulltext, rng, used):
    """Prefer method/result sections; fall back to any non-front-matter section so a
    paper is never silently unusable just because its headings are unusual."""
    for strict in (True, False):
        pool = []
        for section in usable_sections(fulltext, strict=strict):
            for sentence in SENTENCE.split(section.text):
                cleaned = ' '.join(sentence.split())
                if MIN_EVIDENCE <= len(cleaned) <= MAX_EVIDENCE and cleaned not in used:
                    pool.append((section.name, cleaned))
        if pool:
            return rng.choice(pool)
    return None


def write_question(llm, evidence):
    response = llm.complete('Sentence: ' + evidence, system=QUESTION_SYSTEM,
                            temperature=0.0, max_tokens=120)
    return ' '.join((response.text or '').split())


def is_refusal(llm, answer):
    if not (answer or '').strip():
        return True
    response = llm.complete('Answer: ' + answer[:1500], system=ABSTENTION_SYSTEM,
                            temperature=0.0, max_tokens=4)
    return 'REFUSED' in (response.text or '').upper()


def evidence_in_contexts(evidence, contexts):
    needle = normalize(evidence)[:120]
    if not needle:
        return False
    for item in contexts:
        record = item.get('record') or {}
        if needle in normalize(record.get('text')):
            return True
    return False


def run_arm(name, retriever, questions, llm, top_k, judge):
    hits, refusals, supports, details = [], 0, [], []
    for item in questions:
        result = ask_with_retriever(item['question'], retriever, llm, top_k=top_k)
        found = evidence_in_contexts(item['evidence'], result['contexts'])
        refused = is_refusal(llm, result.get('answer'))
        audit = judge(result, llm, max_claims=2)
        hits.append(1.0 if found else 0.0)
        refusals += 1 if refused else 0
        if audit['pairs']:
            supports.append(audit['citation_support_rate'])
        details.append({'question': item['question'], 'evidence_found': found,
                        'refused': refused, 'citation_support': audit['citation_support_rate'],
                        'answer_chars': len(result.get('answer') or '')})
    n = max(1, len(questions))
    return {
        'arm': name,
        'questions': len(questions),
        'evidence_recall': round(sum(hits) / n, 4),
        'refusal_rate': round(refusals / n, 4),
        'citation_support_rate': round(sum(supports) / len(supports), 4) if supports else 0.0,
        'details': details,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description='Abstract vs full-text quantified comparison')
    parser.add_argument('--topic', default='graph neural network')
    parser.add_argument('--papers', type=int, default=6)
    parser.add_argument('--questions', type=int, default=6)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--candidates', type=int, default=4)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--cache-dir', default='')
    parser.add_argument('--out', default='metrics_fulltext.json')
    args = parser.parse_args(argv)

    from eval.judge import audit_answer
    llm = build_llm()
    rng = random.Random(args.seed)

    pooled = []
    for source in build_sources(['arxiv', 'crossref', 'pubmed']):
        try:
            pooled.extend(source.search(args.topic, args.papers))
        except Exception as exc:
            print('[source] %s failed: %r' % (source.name, exc), file=sys.stderr)
    papers = dedupe(pooled)
    print('[corpus] %d papers' % len(papers), file=sys.stderr)

    fulltexts = load_many(papers, cache_dir=args.cache_dir or None)
    usable = [p for p in papers if fulltexts[p.uid].ok]
    print('[fulltext] %d/%d papers usable' % (len(usable), len(papers)), file=sys.stderr)
    if not usable:
        print('no full text for this topic; nothing to compare', file=sys.stderr)
        return 2

    embedder = build_embedder()
    abstract_store = build_index(papers, embedder)
    fulltext_store, stats = build_fulltext_index(papers, embedder, fulltexts)
    print('[index] abstract=%d chunks  fulltext=%d chunks' % (len(abstract_store), stats['chunks']),
          file=sys.stderr)

    questions, used, guard = [], set(), 0
    order = list(usable)
    while len(questions) < args.questions and guard < args.questions * 10:
        paper = order[guard % len(order)]
        guard += 1
        picked = pick_evidence(fulltexts[paper.uid], rng, used)
        if not picked:
            continue
        section, evidence = picked
        used.add(evidence)
        question = write_question(llm, evidence)
        if not question:
            continue
        questions.append({'paper_uid': paper.uid, 'section': section,
                          'evidence': evidence, 'question': question})
        print('[q%d] %s   (%s / %s)' % (len(questions), question[:64], paper.uid, section),
              file=sys.stderr)

    if not questions:
        print('could not build questions', file=sys.stderr)
        return 2

    abstract_arm = run_arm('abstract-only (dense)',
                           DenseRetriever(abstract_store, embedder, name='abstract-dense'),
                           questions, llm, args.top_k, audit_answer)
    two_stage = TwoStageRetriever(abstract_store, fulltext_store, embedder,
                                  candidates=args.candidates, name='two-stage-fulltext')
    fulltext_arm = run_arm('two-stage full-text', two_stage, questions, llm,
                           args.top_k, audit_answer)

    report = {'topic': args.topic, 'corpus_papers': len(papers),
              'papers_with_fulltext': len(usable), 'abstract_chunks': len(abstract_store),
              'fulltext_chunks': stats['chunks'], 'questions': questions,
              'results': [abstract_arm, fulltext_arm]}
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    header = '%-24s %-16s %-14s %s' % ('arm', 'evidence_recall', 'refusal_rate', 'citation_support')
    print('')
    print(header)
    print('-' * len(header))
    for row in report['results']:
        print('%-24s %-16s %-14s %s' % (row['arm'], row['evidence_recall'],
                                        row['refusal_rate'], row['citation_support_rate']))
    print('')
    print('written to ' + args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
