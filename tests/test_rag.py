import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.index import HashingEmbedder
from app.index.pipeline import build_index
from app.llm import ScriptedClient
from app.models import Paper
from app.rag import ask, build_messages, extract_markers, resolve_citations, split_claims


def make_store():
    embedder = HashingEmbedder(dim=256)
    store = build_index([
        Paper(source='arxiv', source_id='1', title='Graph Neural Networks',
              abstract='Graph neural networks model relational data with message passing.'),
        Paper(source='arxiv', source_id='2', title='Byzantine Poetry',
              abstract='Manuscript transmission in the middle ages.'),
    ], embedder)
    return store, embedder


class TestCitations(unittest.TestCase):
    def test_extract_markers_dedupes_and_keeps_order(self):
        self.assertEqual(extract_markers('A [2] then [1] then [2]'), [2, 1])

    def test_extract_markers_ignores_plain_brackets(self):
        self.assertEqual(extract_markers('see [abc] and [12]'), [12])

    def test_split_claims_keeps_only_cited_sentences(self):
        text = 'Cited one [1]. Not cited here. Cited two [2].'
        self.assertEqual(split_claims(text), ['Cited one [1].', 'Cited two [2].'])

    def test_split_claims_handles_chinese_punctuation(self):
        text = '这是结论一[1]。这是没有引用的句子。这是结论二[2]。'
        self.assertEqual(len(split_claims(text)), 2)

    def test_resolve_citations_flags_unknown_marker(self):
        contexts = [{'n': 1, 'score': 0.5, 'record': {'paper_uid': 'arxiv:1', 'uid': 'arxiv:1#0', 'chunk_index': 0, 'text': 'x'}}]
        result = resolve_citations('Good [1] and bad [7].', contexts)
        self.assertEqual(result['used'], [1, 7])
        self.assertEqual(len(result['valid']), 1)
        self.assertEqual(result['invalid'], [7])
        self.assertEqual(result['valid'][0]['paper_uid'], 'arxiv:1')


class TestPrompt(unittest.TestCase):
    def test_prompt_contains_sources_and_question(self):
        contexts = [{'n': 1, 'score': 0.9, 'record': {'paper_uid': 'arxiv:1', 'text': 'chunk body'}}]
        messages = build_messages('What is a GNN?', contexts)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]['role'], 'system')
        user = messages[1]['content']
        self.assertIn('[1] arxiv:1', user)
        self.assertIn('chunk body', user)
        self.assertIn('What is a GNN?', user)


class TestAskPipeline(unittest.TestCase):
    def test_ask_maps_citations_back_to_chunks(self):
        store, embedder = make_store()
        llm = ScriptedClient(['Graph neural networks use message passing [1].'])
        result = ask('how do graph neural networks work', store, embedder, llm, top_k=1)
        self.assertFalse(result['empty'])
        self.assertEqual(result['citations']['used'], [1])
        self.assertEqual(result['citations']['invalid'], [])
        self.assertEqual(result['citations']['valid'][0]['paper_uid'], 'arxiv:1')
        self.assertEqual(len(result['citations']['claims']), 1)
        self.assertEqual(len(llm.calls), 1)

    def test_ask_with_empty_store_skips_llm(self):
        from app.index.store import VectorStore
        llm = ScriptedClient(['should not be used'])
        result = ask('anything', VectorStore(), HashingEmbedder(dim=64), llm)
        self.assertTrue(result['empty'])
        self.assertEqual(llm.calls, [])
        self.assertEqual(result['answer'], '')

    def test_answer_with_unbacked_citation_is_flagged(self):
        store, embedder = make_store()
        llm = ScriptedClient(['Made up claim [9].'])
        result = ask('anything about gnn', store, embedder, llm, top_k=2)
        self.assertEqual(result['citations']['invalid'], [9])
        self.assertEqual(result['citations']['valid'], [])
