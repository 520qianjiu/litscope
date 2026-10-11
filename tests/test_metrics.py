import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import mean, percentile, recall_at_k, reciprocal_rank


class TestMetrics(unittest.TestCase):
    def test_recall_at_k_hits(self):
        ranked = ['a', 'b', 'c', 'd']
        self.assertEqual(recall_at_k(ranked, ['c'], 3), 1.0)
        self.assertEqual(recall_at_k(ranked, ['d'], 3), 0.0)
        self.assertEqual(recall_at_k(ranked, ['d'], 4), 1.0)

    def test_recall_partial_for_multiple_expected(self):
        self.assertAlmostEqual(recall_at_k(['a', 'b', 'c'], ['a', 'c'], 2), 0.5)

    def test_recall_without_labels(self):
        self.assertEqual(recall_at_k(['a'], [], 5), 0.0)

    def test_reciprocal_rank(self):
        self.assertEqual(reciprocal_rank(['a', 'b', 'c'], ['c']), 1.0 / 3.0)
        self.assertEqual(reciprocal_rank(['a', 'b'], ['b']), 0.5)
        self.assertEqual(reciprocal_rank(['a', 'b'], ['z']), 0.0)
        self.assertEqual(reciprocal_rank([], ['a']), 0.0)

    def test_mean(self):
        self.assertEqual(mean([1, 2, 3]), 2.0)
        self.assertEqual(mean([]), 0.0)

    def test_percentile_uses_linear_interpolation(self):
        # same convention as numpy.percentile
        self.assertEqual(percentile([1, 2, 3, 4], 0.5), 2.5)
        self.assertEqual(percentile([1, 2, 3, 4], 0.0), 1.0)
        self.assertEqual(percentile([1, 2, 3, 4], 1.0), 4.0)
        self.assertEqual(percentile([5], 0.9), 5.0)
        self.assertEqual(percentile([], 0.5), 0.0)
