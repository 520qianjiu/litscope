import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.index import HashingEmbedder, VectorStore, chunk_text, cosine, split_sentences
from app.index.pipeline import build_index
from app.models import Paper


class TestChunker(unittest.TestCase):
    def test_split_sentences(self):
        self.assertEqual(split_sentences('One. Two! Three?'),
                         ['One.', 'Two!', 'Three?'])

    def test_blank_input(self):
        self.assertEqual(chunk_text('   '), [])

    def test_respects_max_chars(self):
        text = ' '.join(['Sentence number ' + str(i) + ' is here.' for i in range(40)])
        chunks = chunk_text(text, max_chars=120, overlap_sentences=1)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 200)

    def test_overlap_repeats_boundary_sentence(self):
        text = 'Alpha one. Beta two. Gamma three. Delta four.'
        chunks = chunk_text(text, max_chars=22, overlap_sentences=1)
        self.assertGreater(len(chunks), 1)
        self.assertIn(chunks[0].split('. ')[-1], chunks[1])


class TestEmbedder(unittest.TestCase):
    def test_deterministic_and_normalised(self):
        embedder = HashingEmbedder(dim=64)
        first = embedder.embed_one('graph neural network')
        second = embedder.embed_one('graph neural network')
        self.assertEqual(first, second)
        norm = sum(value * value for value in first) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=6)

    def test_similar_text_scores_higher(self):
        embedder = HashingEmbedder(dim=256)
        query = embedder.embed_one('graph neural network for molecules')
        near = embedder.embed_one('graph neural networks applied to molecules')
        far = embedder.embed_one('medieval poetry and manuscript transmission')
        self.assertGreater(cosine(query, near), cosine(query, far))


class TestStore(unittest.TestCase):
    def setUp(self):
        self.embedder = HashingEmbedder(dim=256)
        self.store = build_index([
            Paper(source='arxiv', source_id='1', title='Graph Neural Networks',
                  abstract='Graph neural networks model relational data.'),
            Paper(source='arxiv', source_id='2', title='Byzantine Poetry',
                  abstract='Manuscript transmission in the middle ages.'),
        ], self.embedder)

    def test_index_size(self):
        self.assertGreaterEqual(len(self.store), 2)

    def test_search_returns_matching_paper_first(self):
        vector = self.embedder.embed_one('graph neural networks relational data')
        hits = self.store.search(vector, top_k=1)
        self.assertEqual(hits[0][1]['paper_uid'], 'arxiv:1')

    def test_save_and_load_round_trip(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'store.json'
            self.store.save(path)
            restored = VectorStore.load(path)
            self.assertEqual(len(restored), len(self.store))
            vector = self.embedder.embed_one('graph neural networks')
            self.assertEqual(restored.search(vector, 1)[0][1]['paper_uid'],
                             self.store.search(vector, 1)[0][1]['paper_uid'])

    def test_cosine_edge_cases(self):
        self.assertEqual(cosine([], [1.0]), 0.0)
        self.assertEqual(cosine([0.0, 0.0], [1.0, 1.0]), 0.0)
