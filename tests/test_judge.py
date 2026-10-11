import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.llm import ScriptedClient
from eval.judge import audit_answer, collect_pairs, parse_verdict


class TestParseVerdict(unittest.TestCase):
    def test_unsupported_wins_over_supported_substring(self):
        # UNSUPPORTED contains SUPPORTED as a substring - order matters
        self.assertEqual(parse_verdict('UNSUPPORTED'), 'UNSUPPORTED')
        self.assertEqual(parse_verdict('unsupported'), 'UNSUPPORTED')
        self.assertEqual(parse_verdict('SUPPORTED'), 'SUPPORTED')

    def test_garbage_is_unclear(self):
        self.assertEqual(parse_verdict('maybe??'), 'UNCLEAR')
        self.assertEqual(parse_verdict(''), 'UNCLEAR')


def make_result(markers, valid_numbers=(1,)):
    valid = [{'n': n, 'paper_uid': 'arxiv:' + str(n), 'chunk_index': 0,
              'text': 'source text for ' + str(n), 'score': 0.5} for n in valid_numbers]
    claims = [{'claim': 'claim ' + str(n) + ' [' + str(n) + '].', 'markers': [n]} for n in markers]
    return {'citations': {'used': list(markers), 'valid': valid, 'invalid': [], 'claims': claims}}


class TestAudit(unittest.TestCase):
    def test_collect_pairs_respects_max_claims(self):
        result = make_result([1, 1, 1])
        self.assertEqual(len(collect_pairs(result, 0)), 3)
        self.assertEqual(len(collect_pairs(result, 2)), 2)

    def test_collect_pairs_skips_unmappable_markers(self):
        result = make_result([1, 9], valid_numbers=(1, 9))
        result['citations']['valid'] = [row for row in result['citations']['valid'] if row['n'] == 1]
        self.assertEqual(len(collect_pairs(result, 0)), 1)

    def test_audit_computes_support_rate(self):
        result = make_result([1, 1])
        llm = ScriptedClient(['SUPPORTED', 'UNSUPPORTED'])
        audit = audit_answer(result, llm, max_claims=2)
        self.assertEqual(audit['pairs'], 2)
        self.assertEqual(audit['supported'], 1)
        self.assertEqual(audit['citation_support_rate'], 0.5)

    def test_audit_without_pairs(self):
        result = {'citations': {'used': [], 'valid': [], 'invalid': [], 'claims': []}}
        audit = audit_answer(result, ScriptedClient([]))
        self.assertEqual(audit['pairs'], 0)
        self.assertEqual(audit['citation_support_rate'], 0.0)
