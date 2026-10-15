import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from app.api import create_app
from app.api.service import CorpusService
from app.index import HashingEmbedder
from app.llm import ScriptedClient
from app.models import Paper
from app.sources.base import Source


class FakeSource(Source):
    name = 'fake'

    def __init__(self, papers=None, error=None):
        self.papers = list(papers or [])
        self.error = error
        self.calls = []

    def search(self, query, limit=10):
        self.calls.append((query, limit))
        if self.error:
            raise RuntimeError(self.error)
        return list(self.papers[:limit])


def sample_papers():
    return [
        Paper(source='arxiv', source_id='1', title='Graph Neural Networks',
              abstract='Graph neural networks propagate messages along edges.'),
        Paper(source='arxiv', source_id='2', title='Graph Neural Network Training Systems',
              abstract='Systems for training graph neural networks at scale.'),
    ]


def make_client(papers=None, replies=None, error=None):
    source = FakeSource(papers if papers is not None else sample_papers(), error=error)
    service = CorpusService(
        llm=ScriptedClient(replies or ['Graph neural networks pass messages along edges [1].']),
        embedder=HashingEmbedder(dim=128),
        source_factory=lambda names: [source],
    )
    return TestClient(create_app(service)), source


class TestHealth(unittest.TestCase):
    def test_health_reports_embedder_and_retriever(self):
        client, _ = make_client()
        body = client.get('/api/health').json()
        self.assertEqual(body['status'], 'ok')
        self.assertEqual(body['embedder'], 'hashing')
        self.assertEqual(body['retriever'], 'bm25')
        self.assertEqual(body['llm'], 'ready')


class TestIndexPage(unittest.TestCase):
    def test_root_serves_ui(self):
        client, _ = make_client()
        response = client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('LitScope', response.text)
        self.assertIn('/api/ask', response.text)


class TestSearch(unittest.TestCase):
    def test_search_returns_deduped_papers(self):
        client, source = make_client()
        response = client.get('/api/search', params={'q': 'graph neural network', 'limit': 5})
        body = response.json()
        self.assertEqual(body['count'], 2)
        self.assertEqual(body['papers'][0]['uid'], 'arxiv:1')
        self.assertEqual(source.calls, [('graph neural network', 5)])

    def test_search_rejects_short_query(self):
        client, _ = make_client()
        self.assertEqual(client.get('/api/search', params={'q': 'a'}).status_code, 422)

    def test_search_surfaces_source_error(self):
        client, _ = make_client(error='upstream blew up')
        body = client.get('/api/search', params={'q': 'anything'}).json()
        self.assertEqual(body['count'], 0)
        self.assertEqual(body['errors'][0]['source'], 'fake')


class TestAsk(unittest.TestCase):
    def test_ask_returns_answer_and_citations(self):
        client, _ = make_client()
        response = client.post('/api/ask', json={
            'topic': 'gnn', 'question': 'graph neural networks messages'})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body['citations']['invalid'], [])
        self.assertEqual(len(body['citations']['valid']), 1)
        self.assertEqual(body['citations']['valid'][0]['paper_uid'], 'arxiv:1')
        self.assertGreater(body['corpus']['chunks'], 0)

    def test_ask_rejects_blank_question(self):
        client, _ = make_client()
        self.assertEqual(client.post('/api/ask', json={
            'topic': 'gnn', 'question': '   '}).status_code, 400)

    def test_ask_with_no_lexical_overlap_returns_empty(self):
        # a Chinese question against an English-only corpus has zero token overlap
        client, _ = make_client()
        body = client.post('/api/ask', json={
            'topic': 'gnn', 'question': '图神经网络怎么传播信息？'}).json()
        self.assertTrue(body['empty'])
        self.assertEqual(body['contexts'], [])
        self.assertEqual(body['answer'], '')

    def test_corpus_is_cached_between_calls(self):
        client, source = make_client(replies=['First [1].', 'Second [1].'])
        client.post('/api/ask', json={'topic': 'gnn', 'question': 'graph neural networks'})
        client.post('/api/ask', json={'topic': 'gnn', 'question': 'graph neural network training'})
        self.assertEqual(len(source.calls), 1)
        self.assertEqual(client.get('/api/health').json()['cached_corpora'], 1)

    def test_different_topic_rebuilds_corpus(self):
        client, source = make_client(replies=['First [1].', 'Second [1].'])
        client.post('/api/ask', json={'topic': 'gnn', 'question': 'graph neural networks'})
        client.post('/api/ask', json={'topic': 'other topic',
                                      'question': 'graph neural network training'})
        self.assertEqual(len(source.calls), 2)


class TestFulltextApiMode(unittest.TestCase):
    """The API must be able to serve full-text retrieval, and must degrade to the
    abstract index when a topic has no downloadable bodies."""

    def make_service(self, fulltext_ok=True):
        from app.api.service import CorpusService
        from app.fulltext import FullText, Section
        from app.index import HashingEmbedder

        class FakeSource(Source):
            name = 'fake'

            def search(self, query, limit=10):
                return [Paper(source='arxiv', source_id='2406.00552v4', title='GNN Training Systems',
                              abstract='Mini-batch and full-graph training differ.'),
                        Paper(source='arxiv', source_id='1909.10086v3', title='Graph Embeddings',
                              abstract='Embeddings are task dependent.')]

        def loader(papers, cache_dir=None):
            out = {}
            for index, paper in enumerate(papers):
                if fulltext_ok and index == 0:
                    out[paper.uid] = FullText(source_id=paper.source_id, sections=[
                        Section(name='EXPERIMENTAL METHODOLOGY',
                                text='We compare against three baselines: DGL, BNS-GCN and full-graph.'),
                        Section(name='PERFORMANCE EVALUATION',
                                text='Mini-batch training reaches lower time-to-accuracy.')])
                else:
                    out[paper.uid] = FullText(source_id=paper.source_id, error='no PMC record (not open access)')
            return out

        return CorpusService(llm=ScriptedClient(['Answer [1].']),
                             embedder=HashingEmbedder(dim=128),
                             source_factory=lambda names: [FakeSource()],
                             retriever='fulltext',
                             fulltext_loader=loader)

    def test_fulltext_mode_builds_a_two_stage_reranked_retriever(self):
        service = self.make_service(fulltext_ok=True)
        entry = service.corpus('gnn')
        self.assertIsNotNone(entry['fulltext_store'])
        retriever = service.retriever_for(entry)
        self.assertIn('fulltext', retriever.name)

    def test_ask_reports_fulltext_stats(self):
        service = self.make_service(fulltext_ok=True)
        result = service.ask('gnn', 'graph neural network training baselines')
        self.assertEqual(result['corpus']['fulltext']['with_fulltext'], 1)
        self.assertGreater(result['corpus']['fulltext']['chunks'], 0)
        self.assertEqual(result['corpus']['reranker'], 'noop')

    def test_falls_back_to_abstract_when_no_fulltext(self):
        service = self.make_service(fulltext_ok=False)
        entry = service.corpus('gnn')
        self.assertIsNone(entry['fulltext_store'])
        retriever = service.retriever_for(entry)
        self.assertNotIn('fulltext', retriever.name)
        result = service.ask('gnn', 'graph neural network training baselines')
        self.assertFalse(result['empty'])

    def test_fulltext_disabled_by_default(self):
        from app.api.service import CorpusService
        from app.index import HashingEmbedder

        class QuietSource(Source):
            name = 'quiet'

            def search(self, query, limit=10):
                return [Paper(source='arxiv', source_id='1', title='T', abstract='Body.')]

        service = CorpusService(embedder=HashingEmbedder(dim=64), retriever='dense',
                                source_factory=lambda names: [QuietSource()])
        self.assertFalse(service.wants_fulltext())
        entry = service.corpus('anything')
        self.assertIsNone(entry['fulltext'])
