"""Cold-start behaviour: parallel embedding batches and parallel full-text fetches.

These tests run fully offline: the network layer is replaced by a tracking
stand-in, so they assert concurrency structure (order preserved, bounded
in-flight requests, tolerant env parsing) rather than provider latency.
"""

import json
import os
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.fulltext import FullText, Section, load_many
from app.index.embedder import OpenAICompatEmbedder
from app.models import Paper


class TrackingEmbedder(OpenAICompatEmbedder):
    """Offline stand-in: echoes the input index so vector order is verifiable."""

    def __init__(self, delay=0.02):
        super().__init__('http://localhost:1', 'test-key', 'test-model', dim=1, batch_size=2)
        self.delay = delay
        self.lock = threading.Lock()
        self.calls = 0
        self.in_flight = 0
        self.max_in_flight = 0

    def _embed_batch(self, batch, attempts=3):
        with self.lock:
            self.calls += 1
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            time.sleep(self.delay)
            return [[float(item)] for item in batch]
        finally:
            with self.lock:
                self.in_flight -= 1


class TestParallelEmbedding(unittest.TestCase):
    def setUp(self):
        self.saved = {k: os.environ.get(k) for k in ('EMBEDDING_BATCH', 'EMBEDDING_CONCURRENCY')}
        os.environ['EMBEDDING_BATCH'] = '2'

    def tearDown(self):
        for key, value in self.saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_order_is_preserved_with_parallel_batches(self):
        os.environ['EMBEDDING_CONCURRENCY'] = '4'
        embedder = TrackingEmbedder()
        vectors = embedder.embed([str(i) for i in range(10)])
        self.assertEqual(vectors, [[float(i)] for i in range(10)])
        self.assertEqual(embedder.calls, 5)
        self.assertGreater(embedder.max_in_flight, 1)

    def test_concurrency_one_stays_serial(self):
        os.environ['EMBEDDING_CONCURRENCY'] = '1'
        embedder = TrackingEmbedder()
        vectors = embedder.embed([str(i) for i in range(6)])
        self.assertEqual(vectors, [[float(i)] for i in range(6)])
        self.assertEqual(embedder.max_in_flight, 1)

    def test_garbage_concurrency_falls_back_instead_of_crashing(self):
        # This class of env mistake (a model name where a number belongs) already
        # took one deploy down once; it must degrade, not raise.
        os.environ['EMBEDDING_CONCURRENCY'] = 'BAAI/bge-m3'
        embedder = TrackingEmbedder()
        vectors = embedder.embed([str(i) for i in range(6)])
        self.assertEqual(vectors, [[float(i)] for i in range(6)])

    def test_garbage_batch_size_falls_back_instead_of_crashing(self):
        os.environ['EMBEDDING_BATCH'] = 'sixty-four'
        embedder = TrackingEmbedder()
        vectors = embedder.embed([str(i) for i in range(9)])
        self.assertEqual(vectors, [[float(i)] for i in range(9)])


class TestEmbeddingRetry(unittest.TestCase):
    def test_transient_network_error_is_retried(self):
        # Regression guard: the retry path called time.sleep without importing
        # time, so the first transient failure crashed with NameError instead of
        # retrying.
        embedder = OpenAICompatEmbedder('http://localhost:1', 'k', 'm', dim=1, batch_size=2)
        attempts = []

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return json.dumps({'data': [{'index': 0, 'embedding': [7.0]}]}).encode()

        def flaky(request, timeout=None):
            attempts.append(1)
            if len(attempts) < 2:
                raise ConnectionResetError('transient')
            return FakeResponse()

        with mock.patch('urllib.request.urlopen', side_effect=flaky), \
                mock.patch('time.sleep', return_value=None):
            vectors = embedder._embed_batch(['x'])
        self.assertEqual(vectors, [[7.0]])
        self.assertEqual(len(attempts), 2)


class TestParallelFulltextFetch(unittest.TestCase):
    def setUp(self):
        self.saved = os.environ.get('LITSCOPE_FULLTEXT_CONCURRENCY')
        self.state = {'lock': threading.Lock(), 'in_flight': 0, 'max_in_flight': 0}

    def tearDown(self):
        if self.saved is None:
            os.environ.pop('LITSCOPE_FULLTEXT_CONCURRENCY', None)
        else:
            os.environ['LITSCOPE_FULLTEXT_CONCURRENCY'] = self.saved

    def fake_load(self, paper, cache_dir=None):
        with self.state['lock']:
            self.state['in_flight'] += 1
            self.state['max_in_flight'] = max(self.state['max_in_flight'],
                                              self.state['in_flight'])
        try:
            time.sleep(0.02)
            return FullText(source_id=paper.source_id, sections=[Section(name='INTRO', text='t')])
        finally:
            with self.state['lock']:
                self.state['in_flight'] -= 1

    def papers(self, count=6):
        return [Paper(source='arxiv', source_id=str(i), title='t' + str(i), abstract='a')
                for i in range(count)]

    def test_fetch_is_parallel_and_order_is_kept(self):
        os.environ['LITSCOPE_FULLTEXT_CONCURRENCY'] = '3'
        papers = self.papers()
        with mock.patch('app.fulltext.load_fulltext', side_effect=self.fake_load):
            out = load_many(papers)
        self.assertEqual(list(out.keys()), [p.uid for p in papers])
        self.assertTrue(all(value.ok for value in out.values()))
        self.assertGreater(self.state['max_in_flight'], 1)

    def test_concurrency_one_stays_serial(self):
        os.environ['LITSCOPE_FULLTEXT_CONCURRENCY'] = '1'
        with mock.patch('app.fulltext.load_fulltext', side_effect=self.fake_load):
            out = load_many(self.papers(4))
        self.assertEqual(len(out), 4)
        self.assertEqual(self.state['max_in_flight'], 1)

    def test_garbage_value_falls_back(self):
        os.environ['LITSCOPE_FULLTEXT_CONCURRENCY'] = 'fast please'
        with mock.patch('app.fulltext.load_fulltext', side_effect=self.fake_load):
            out = load_many(self.papers(4))
        self.assertEqual(len(out), 4)


class TestAskTimings(unittest.TestCase):
    def test_ask_reports_stage_timings(self):
        from app.api.service import CorpusService
        from app.index import HashingEmbedder
        from app.llm import ScriptedClient
        from app.sources.base import Source

        class FakeSource(Source):
            name = 'fake'

            def search(self, query, limit=10):
                return [Paper(source='arxiv', source_id='1', title='Graph Neural Networks',
                              abstract='Graph neural networks propagate messages along edges.')]

        service = CorpusService(
            llm=ScriptedClient(['Graph neural networks pass messages along edges [1].']),
            embedder=HashingEmbedder(dim=64),
            source_factory=lambda names: [FakeSource()],
            retriever='bm25',
        )
        result = service.ask('graph neural network', 'how do they pass messages?')
        self.assertIn('timings', result)
        self.assertIn('sources', result['timings'])
        self.assertIn('abstract_index', result['timings'])
        self.assertIn('build_total', result['timings'])
        self.assertIn('corpus_wait', result['timings'])
        self.assertIn('answer', result['timings'])
        # The second ask is served from the in-memory corpus cache, so the wait
        # for the corpus should be ~0 while the build timings are still reported.
        again = service.ask('graph neural network', 'how do they pass messages?')
        self.assertLessEqual(again['timings']['corpus_wait'], 0.5)
        self.assertIn('build_total', again['timings'])