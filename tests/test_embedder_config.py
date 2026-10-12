import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.index.embedder import HashingEmbedder, build_embedder


class TestTolerantEnvParsing(unittest.TestCase):
    def setUp(self):
        self.saved = {k: os.environ.get(k) for k in
                      ('EMBEDDING_DIM', 'EMBEDDING_NGRAM', 'EMBEDDING_PROVIDER')}

    def tearDown(self):
        for key, value in self.saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_model_name_in_dim_does_not_crash(self):
        # this is the exact value that crashed the Railway deploy at import time
        os.environ['EMBEDDING_DIM'] = 'BAAI/bge-m3'
        embedder = build_embedder('hashing')
        self.assertIsInstance(embedder, HashingEmbedder)
        self.assertEqual(embedder.dim, 256)

    def test_other_garbage_values_fall_back(self):
        for bad in ['', '  ', 'abc', '10.5', '-8', '0']:
            os.environ['EMBEDDING_DIM'] = bad
            self.assertEqual(build_embedder('hashing').dim, 256, msg='failed for ' + repr(bad))

    def test_valid_dim_is_respected(self):
        os.environ['EMBEDDING_DIM'] = '128'
        self.assertEqual(build_embedder('hashing').dim, 128)

    def test_bad_ngram_falls_back(self):
        os.environ['EMBEDDING_NGRAM'] = 'BAAI/bge-m3'
        self.assertEqual(build_embedder('hashing').ngram, 3)

    def test_missing_embedding_keys_lists_names(self):
        os.environ['EMBEDDING_PROVIDER'] = 'openai'
        for key in ('EMBEDDING_BASE_URL', 'EMBEDDING_API_KEY', 'EMBEDDING_MODEL'):
            os.environ.pop(key, None)
        with self.assertRaises(ValueError) as ctx:
            build_embedder()
        message = str(ctx.exception)
        self.assertIn('EMBEDDING_BASE_URL', message)
        self.assertIn('EMBEDDING_API_KEY', message)
        self.assertIn('EMBEDDING_MODEL', message)


class TestServiceSurvivesBadConfig(unittest.TestCase):
    def setUp(self):
        self.saved = {k: os.environ.get(k) for k in
                      ('EMBEDDING_PROVIDER', 'EMBEDDING_DIM', 'EMBEDDING_BASE_URL')}

    def tearDown(self):
        for key, value in self.saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_service_starts_when_embedding_config_is_broken(self):
        from app.api.service import CorpusService
        os.environ['EMBEDDING_PROVIDER'] = 'openai'
        os.environ['EMBEDDING_DIM'] = 'BAAI/bge-m3'
        os.environ.pop('EMBEDDING_BASE_URL', None)
        service = CorpusService()
        self.assertTrue(service.embedder_error)
        self.assertEqual(service.embedder.name, 'hashing')
        report = service.config_report()
        self.assertIn('EMBEDDING_BASE_URL', report)
        self.assertEqual(report['EMBEDDING_BASE_URL'], 'MISSING')
