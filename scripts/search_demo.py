import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.dedupe import dedupe
from app.sources import build_sources


def main(argv=None):
    parser = argparse.ArgumentParser(description='LitScope W1: multi-source search + dedupe')
    parser.add_argument('query')
    parser.add_argument('--limit', type=int, default=5)
    parser.add_argument('--sources', default='arxiv,crossref,pubmed')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args(argv)

    names = [n.strip() for n in args.sources.split(',') if n.strip()]
    raw = []
    errors = []
    for source in build_sources(names):
        try:
            found = source.search(args.query, args.limit)
            print('[ok]   ' + source.name + ' -> ' + str(len(found)), file=sys.stderr)
            raw.extend(found)
        except Exception as exc:
            errors.append((source.name, repr(exc)))
            print('[fail] ' + source.name + ' -> ' + repr(exc), file=sys.stderr)

    papers = dedupe(raw)
    if args.json:
        print(json.dumps([p.to_dict() for p in papers], ensure_ascii=False, indent=2))
    else:
        print('')
        print('query: ' + args.query)
        print('raw hits: ' + str(len(raw)) + '   after dedupe: ' + str(len(papers)))
        for index, paper in enumerate(papers, 1):
            print('[' + str(index) + '] ' + (paper.title or '(no title)'))
            print('    ' + paper.source + ':' + paper.source_id
                  + '   year=' + str(paper.year) + '   cites=' + str(paper.citations))
            if paper.doi:
                print('    doi: ' + paper.doi)
            if paper.merged_sources:
                print('    merged: ' + ', '.join(paper.merged_sources))
    if errors and not papers:
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
