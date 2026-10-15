import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.fulltext import (FullText, Section, arxiv_id, is_heading, pdf_url,
                          split_sections)
from app.index import HashingEmbedder
from app.index.chunker import fulltext_to_chunks, sections_to_chunks
from app.index.pipeline import build_fulltext_index
from app.index.retrievers import TwoStageRetriever
from app.index.store import VectorStore
from app.models import Paper


def sample_text():
    return '\n'.join([
        'Some Title Here',
        'Author One, Author Two',
        'ABSTRACT',
        'We study graph training.',
        'INTRODUCTION',
        'Graphs are everywhere. This section is long enough to chunk.',
        'EXPERIMENTAL METHODOLOGY',
        'We trained with a learning rate of 0.001 for 100 epochs.',
        'CONCLUSION',
        'Mini-batch training is faster.',
    ])


class TestHeadingDetection(unittest.TestCase):
    def test_accepts_real_headings(self):
        for line in ['INTRODUCTION', 'BACKGROUND AND MOTIVATION',
                     'EXPERIMENTAL METHODOLOGY', 'PERFORMANCE EVALUATION',
                     'COST ANALYSIS', 'RELATED WORK', 'REFERENCES', 'APPENDIX']:
            self.assertTrue(is_heading(line), msg=line)

    def test_rejects_table_cells_and_acronyms(self):
        for line in ['GPU', 'ACC', 'DGL', 'TTA', 'BNS-GCN', 'ACC, TTA',
                     '3.556 924.46', 'ROC [20]', '']:
            self.assertFalse(is_heading(line), msg=line)

    def test_rejects_mixed_case_and_long_lines(self):
        self.assertFalse(is_heading('Introduction'))
        self.assertFalse(is_heading('A ' * 40))

    def test_rejects_headings_with_digits(self):
        self.assertFalse(is_heading('SECTION 1 INTRODUCTION'))


class TestSplitSections(unittest.TestCase):
    def test_splits_into_named_sections(self):
        sections = split_sections(sample_text())
        names = [s.name for s in sections]
        self.assertEqual(names, ['FRONT MATTER', 'ABSTRACT', 'INTRODUCTION',
                                 'EXPERIMENTAL METHODOLOGY', 'CONCLUSION'])

    def test_front_matter_is_kept_not_dropped(self):
        sections = split_sections(sample_text())
        self.assertIn('Author One', sections[0].text)

    def test_text_without_headings_becomes_one_section(self):
        sections = split_sections('just some body text with no headings at all')
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0].name, 'FULLTEXT')

    def test_empty_input(self):
        self.assertEqual(split_sections(''), [])

    def test_full_coverage(self):
        text = sample_text()
        sections = split_sections(text)
        covered = sum(s.char_count for s in sections)
        self.assertGreater(covered / len(text), 0.95)


class TestArxivId(unittest.TestCase):
    def test_modern_id_with_version(self):
        paper = Paper(source='arxiv', source_id='2406.00552v4', title='t',
                      url='http://arxiv.org/abs/2406.00552v4')
        self.assertEqual(arxiv_id(paper), '2406.00552v4')
        self.assertEqual(pdf_url(paper), 'https://arxiv.org/pdf/2406.00552v4')

    def test_legacy_id(self):
        paper = Paper(source='arxiv', source_id='cs/0701001v1', title='t',
                      url='http://arxiv.org/abs/cs/0701001v1')
        self.assertEqual(arxiv_id(paper), 'cs/0701001v1')

    def test_non_arxiv_paper_has_no_url(self):
        paper = Paper(source='pubmed', source_id='12345', title='t')
        self.assertIsNone(arxiv_id(paper))
        self.assertIsNone(pdf_url(paper))


class TestSectionChunking(unittest.TestCase):
    def setUp(self):
        self.paper = Paper(source='arxiv', source_id='1', title='T')
        self.fulltext = FullText(source_id='1', sections=[
            Section(name='INTRODUCTION', text='Alpha beta. ' * 40),
            Section(name='RESULTS', text='Gamma delta. ' * 40),
        ])

    def test_every_chunk_carries_its_section(self):
        chunks = sections_to_chunks('arxiv:1', self.fulltext.sections, max_chars=120)
        self.assertTrue(chunks)
        self.assertTrue(all(c.section for c in chunks))
        self.assertEqual({c.section for c in chunks}, {'INTRODUCTION', 'RESULTS'})

    def test_chunk_index_is_continuous_across_sections(self):
        chunks = fulltext_to_chunks(self.paper, self.fulltext, max_chars=120)
        self.assertEqual([c.chunk_index for c in chunks], list(range(len(chunks))))

    def test_no_chunks_for_empty_sections(self):
        self.assertEqual(sections_to_chunks('a', [Section(name='X', text='')]), [])


class TestFulltextIndex(unittest.TestCase):
    def setUp(self):
        self.embedder = HashingEmbedder(dim=128)
        self.papers = [Paper(source='arxiv', source_id='1', title='One'),
                       Paper(source='arxiv', source_id='2', title='Two')]

    def test_reports_skipped_papers_with_reasons(self):
        good = FullText(source_id='1', sections=[Section(name='ABSTRACT', text='Body text here.')])
        bad = FullText(source_id='2', error='download failed: URLError')
        store, stats = build_fulltext_index(self.papers, self.embedder,
                                            {'arxiv:1': good, 'arxiv:2': bad})
        self.assertEqual(stats['papers_with_fulltext'], 1)
        self.assertEqual(len(stats['skipped']), 1)
        self.assertIn('download failed', stats['skipped'][0]['reason'])
        self.assertEqual(stats['sections'], ['ABSTRACT'])
        self.assertGreater(len(store), 0)

    def test_missing_entry_is_not_counted_as_success(self):
        store, stats = build_fulltext_index(self.papers, self.embedder, {})
        self.assertEqual(stats['papers_with_fulltext'], 0)
        self.assertEqual(len(stats['skipped']), 2)
        self.assertEqual(len(store), 0)


class TestTwoStageRetriever(unittest.TestCase):
    def setUp(self):
        self.embedder = HashingEmbedder(dim=256)
        self.abstracts = VectorStore()
        self.abstracts.add(
            self.embedder.embed(['graph neural network training', 'manuscript transmission middle ages']),
            [{'paper_uid': 'arxiv:1', 'text': 'graph neural network training'},
             {'paper_uid': 'arxiv:2', 'text': 'manuscript transmission middle ages'}])
        self.fulltext = VectorStore()
        self.fulltext.add(
            self.embedder.embed(['graph training mini batch', 'graph training full graph',
                                 'medieval poetry and scribes']),
            [{'paper_uid': 'arxiv:1', 'chunk_index': 0, 'uid': 'arxiv:1#0', 'text': 'graph training mini batch', 'section': 'METHODS'},
             {'paper_uid': 'arxiv:1', 'chunk_index': 1, 'uid': 'arxiv:1#1', 'text': 'graph training full graph', 'section': 'RESULTS'},
             {'paper_uid': 'arxiv:2', 'chunk_index': 0, 'uid': 'arxiv:2#0', 'text': 'medieval poetry and scribes', 'section': 'INTRO'}])

    def test_stage_one_limits_stage_two(self):
        retriever = TwoStageRetriever(self.abstracts, self.fulltext, self.embedder, candidates=1)
        hits = retriever.search('graph neural network training', top_k=5)
        self.assertTrue(hits)
        self.assertEqual({r['paper_uid'] for _s, r in hits}, {'arxiv:1'})
        self.assertEqual(retriever.last_shortlist, ['arxiv:1'])

    def test_no_shortlist_returns_nothing(self):
        empty = VectorStore()
        retriever = TwoStageRetriever(empty, self.fulltext, self.embedder, candidates=2)
        self.assertEqual(retriever.search('anything at all', top_k=3), [])

    def test_chunks_keep_their_section_label(self):
        retriever = TwoStageRetriever(self.abstracts, self.fulltext, self.embedder, candidates=2)
        hits = retriever.search('graph training', top_k=3)
        self.assertTrue(all(r.get('section') for _s, r in hits))


class TestStoreFilter(unittest.TestCase):
    def test_paper_ids_restricts_results(self):
        store = VectorStore()
        emb = HashingEmbedder(dim=64)
        store.add(emb.embed(['alpha text', 'beta text']),
                  [{'paper_uid': 'a', 'text': 'alpha text'}, {'paper_uid': 'b', 'text': 'beta text'}])
        both = store.search(emb.embed_one('alpha text'), 5)
        self.assertEqual(len(both), 2)
        only_b = store.search(emb.embed_one('alpha text'), 5, paper_ids=['b'])
        self.assertEqual([r['paper_uid'] for _s, r in only_b], ['b'])


class TestCacheFailureIsNotFatal(unittest.TestCase):
    def test_unwritable_cache_dir_still_returns_pdf(self):
        from app.fulltext import fetch_pdf
        import urllib.request
        original = urllib.request.urlopen
        payload = b'%PDF-1.4 fake'

        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return payload

        urllib.request.urlopen = lambda *a, **k: FakeResponse()
        try:
            # a path that cannot be created on Windows (illegal characters)
            data = fetch_pdf('https://example.invalid/x.pdf', cache_dir='Z:\\nope\\bad:*?')
            self.assertEqual(data, payload)
        finally:
            urllib.request.urlopen = original


class TestCitationCarriesSection(unittest.TestCase):
    def test_section_label_survives_into_citations(self):
        from app.rag.citations import resolve_citations
        contexts = [{'n': 1, 'score': 0.5,
                     'record': {'paper_uid': 'arxiv:1', 'uid': 'arxiv:1#0', 'chunk_index': 0,
                                'text': 'body', 'section': 'RESULTS'}}]
        result = resolve_citations('The claim holds [1].', contexts)
        self.assertEqual(result['valid'][0]['section'], 'RESULTS')

    def test_missing_section_degrades_to_empty_string(self):
        from app.rag.citations import resolve_citations
        contexts = [{'n': 1, 'score': 0.5,
                     'record': {'paper_uid': 'arxiv:1', 'uid': 'arxiv:1#0', 'chunk_index': 0,
                                'text': 'body'}}]
        result = resolve_citations('The claim holds [1].', contexts)
        self.assertEqual(result['valid'][0]['section'], '')
