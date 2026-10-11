import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.sources.arxiv import ArxivSource


def main(argv=None):
    parser = argparse.ArgumentParser(description='Build LitScope eval set from real arXiv metadata')
    parser.add_argument('--topic', default='language model')
    parser.add_argument('--count', type=int, default=20)
    parser.add_argument('--out', default='eval/eval_set.json')
    args = parser.parse_args(argv)

    papers = ArxivSource().search(args.topic, limit=min(args.count, 100))
    items = []
    for paper in papers:
        items.append({
            'query': paper.title,
            'expected': [paper.uid],
            'target': paper.to_dict(),
        })
    payload = {
        'topic': args.topic,
        'label_scheme': 'title-recovery: query is the paper title, expected is that paper uid',
        'items': items,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print('wrote ' + str(len(items)) + ' items to ' + str(out))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
