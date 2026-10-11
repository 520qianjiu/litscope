import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.index import BM25Index, HashingEmbedder
from app.index.pipeline import build_index
from app.index.retrievers import BM25Retriever
from app.llm import ScriptedClient
from app.models import Paper
from app.rag import ask_with_retriever


def english_corpus():
    return build_index([
        Paper(source='arxiv', source_id='1', title='Graph Neural Networks for Proteins',
              abstract='Graph neural networks predict protein function from structure.'),
        Paper(source='arxiv', source_id='2', title='Byzantine Poetry',
              abstract='Manuscript transmission in the middle ages and scribal culture.'),
    ], HashingEmbedder(dim=64))


class TestZeroScoreFilter(unittest.TestCase):
    def test_chinese_query_over_english_corpus_returns_nothing(self):
        retriever = BM25Retriever.from_store(english_corpus())
        self.assertEqual(retriever.search('图神经网络在蛋白质功能预测中的应用', 5), [])

    def test_english_query_still_works(self):
        retriever = BM25Retriever.from_store(english_corpus())
        hits = retriever.search('graph neural networks protein function', 5)
        self.assertTrue(hits)
        self.assertEqual(hits[0][1]['paper_uid'], 'arxiv:1')

    def test_all_zero_scores_are_dropped(self):
        index = BM25Index([
            {'uid': 'a', 'text': 'alpha beta'},
            {'uid': 'b', 'text': 'gamma delta'},
        ])
        self.assertEqual(index.search('zzz', 5), [])

    def test_raw_index_can_still_return_zero_scored_rows(self):
        index = BM25Index([{'uid': 'a', 'text': 'alpha beta'}])
        self.assertEqual(len(index.search('zzz', 5, min_score=-1.0)), 1)


class TestDegenerateRetrievalIsHonest(unittest.TestCase):
    def test_pipeline_reports_empty_instead_of_garbage(self):
        store = english_corpus()
        retriever = BM25Retriever.from_store(store)
        llm = ScriptedClient(['should never be called'])
        result = ask_with_retriever('图神经网络怎么用？', retriever, llm, top_k=5)
        self.assertTrue(result['empty'])
        self.assertEqual(result['contexts'], [])
        self.assertEqual(result['answer'], '')
        self.assertEqual(llm.calls, [], 'LLM must not be called when retrieval finds nothing')
