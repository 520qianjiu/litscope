import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.index import BM25Index, HashingEmbedder, build_retriever, tokenize
from app.index.pipeline import build_index as make_index
from app.index.retrievers import BM25Retriever, DenseRetriever
from app.llm import ScriptedClient
from app.models import Paper
from app.rag import ask_with_retriever


def corpus():
    return [
        Paper(source='arxiv', source_id='1', title='Graph Neural Networks',
              abstract='Graph neural networks propagate messages along graph edges.'),
        Paper(source='arxiv', source_id='2', title='Byzantine Poetry',
              abstract='Manuscript transmission in the middle ages and scribal culture.'),
        Paper(source='arxiv', source_id='3', title='Enzyme Kinetics',
              abstract='Enzyme kinetics and substrate specificity measurements.'),
    ]


class TestTokenize(unittest.TestCase):
    def test_lowercases_and_splits(self):
        self.assertEqual(tokenize('Graph, NEURAL networks!'), ['graph', 'neural', 'networks'])

    def test_handles_chinese(self):
        self.assertEqual(tokenize('图神经网络 GNN'), ['图神经网络', 'gnn'])

    def test_empty(self):
        self.assertEqual(tokenize(''), [])


class TestBM25(unittest.TestCase):
    def setUp(self):
        self.index = BM25Index([
            {'uid': 'a', 'text': 'graph neural networks propagate messages'},
            {'uid': 'b', 'text': 'manuscript transmission middle ages'},
            {'uid': 'c', 'text': 'enzyme kinetics substrate specificity'},
        ])

    def test_matching_chunk_is_only_hit(self):
        hits = self.index.search('graph neural networks', 3)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0][1]['uid'], 'a')

    def test_term_absent_from_corpus_yields_no_hits(self):
        self.assertEqual(self.index.search('nonexistentterm', 3), [])

    def test_score_is_per_document(self):
        self.assertGreater(self.index.score('graph', 0), 0.0)
        self.assertEqual(self.index.score('graph', 1), 0.0)

    def test_empty_query_scores_zero(self):
        self.assertEqual(self.index.search('', 3), [])

    def test_top_k_limits_results(self):
        self.assertEqual(len(self.index.search('graph', 1)), 1)


class TestRetrieverFactory(unittest.TestCase):
    def setUp(self):
        self.embedder = HashingEmbedder(dim=64)
        self.store = make_index(corpus(), self.embedder)

    def test_builds_bm25(self):
        retriever = build_retriever('bm25', self.store, self.embedder)
        self.assertIsInstance(retriever, BM25Retriever)
        self.assertEqual(retriever.name, 'bm25')

    def test_builds_dense(self):
        retriever = build_retriever('dense', self.store, self.embedder)
        self.assertIsInstance(retriever, DenseRetriever)
        self.assertEqual(retriever.name, 'dense-hashing')

    def test_rejects_unknown(self):
        with self.assertRaises(ValueError):
            build_retriever('nope', self.store, self.embedder)

    def test_bm25_from_store_finds_own_paper(self):
        retriever = BM25Retriever.from_store(self.store)
        hits = retriever.search('graph neural networks edges', 1)
        self.assertEqual(hits[0][1]['paper_uid'], 'arxiv:1')


class TestAskWithRetriever(unittest.TestCase):
    def test_pipeline_works_with_bm25(self):
        embedder = HashingEmbedder(dim=64)
        store = make_index(corpus(), embedder)
        retriever = BM25Retriever.from_store(store)
        llm = ScriptedClient(['Messages travel along edges [1].'])
        result = ask_with_retriever('graph neural networks messages', retriever, llm, top_k=1)
        self.assertEqual(result['retriever'], 'bm25')
        self.assertEqual(result['citations']['used'], [1])
        self.assertEqual(result['citations']['invalid'], [])
