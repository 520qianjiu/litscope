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


class TestPmcSections(unittest.TestCase):
    def document(self):
        return {'passages': [
            {'infons': {'section_type': 'INTRO', 'type': 'paragraph'}, 'text': 'Opening remark.'},
            {'infons': {'section_type': 'METHODS', 'type': 'paragraph'}, 'text': 'We trained 100 epochs.'},
            {'infons': {'section_type': 'INTRO', 'type': 'paragraph'}, 'text': 'Second intro paragraph.'},
            {'infons': {'section_type': 'RESULTS', 'type': 'paragraph'}, 'text': 'Accuracy improved.'},
        ]}

    def test_groups_passages_by_section_type(self):
        from app.fulltext import bioc_sections
        names = [s.name for s in bioc_sections(self.document())]
        self.assertEqual(names, ['INTRODUCTION', 'METHODS', 'RESULTS'])

    def test_merges_repeated_sections_in_order(self):
        from app.fulltext import bioc_sections
        intro = [s for s in bioc_sections(self.document()) if s.name == 'INTRODUCTION'][0]
        self.assertIn('Opening remark.', intro.text)
        self.assertIn('Second intro paragraph.', intro.text)

    def test_unknown_section_type_is_kept_not_dropped(self):
        from app.fulltext import bioc_sections
        doc = {'passages': [{'infons': {'section_type': 'WEIRD'}, 'text': 'Body.'}]}
        self.assertEqual(bioc_sections(doc)[0].name, 'WEIRD')

    def test_missing_section_type_becomes_fulltext(self):
        from app.fulltext import bioc_sections
        doc = {'passages': [{'infons': {}, 'text': 'Body.'}]}
        self.assertEqual(bioc_sections(doc)[0].name, 'FULLTEXT')

    def test_blank_passages_are_ignored(self):
        from app.fulltext import bioc_sections
        doc = {'passages': [{'infons': {'section_type': 'INTRO'}, 'text': '   '}]}
        self.assertEqual(bioc_sections(doc), [])


class TestPmcFailureReporting(unittest.TestCase):
    def test_paper_without_pmc_record_reports_reason(self):
        import app.fulltext as ft
        original = ft.pmc_id
        ft.pmc_id = lambda *a, **k: None
        try:
            result = ft.load_pmc_fulltext(Paper(source='pubmed', source_id='123', title='t'))
            self.assertFalse(result.ok)
            self.assertIn('no PMC record', result.error)
        finally:
            ft.pmc_id = original

    def test_bioc_failure_reports_reason(self):
        import app.fulltext as ft
        origin_id, origin_doc = ft.pmc_id, ft.bioc_document
        ft.pmc_id = lambda *a, **k: 'PMC1'
        def boom(*a, **k):
            raise RuntimeError('bioc fetch failed: HTTPError')
        ft.bioc_document = boom
        try:
            result = ft.load_pmc_fulltext(Paper(source='pubmed', source_id='123', title='t'))
            self.assertFalse(result.ok)
            self.assertIn('bioc fetch failed', result.error)
        finally:
            ft.pmc_id, ft.bioc_document = origin_id, origin_doc

    def test_successful_pmc_load_returns_sections(self):
        import app.fulltext as ft
        origin_id, origin_doc = ft.pmc_id, ft.bioc_document
        ft.pmc_id = lambda *a, **k: 'PMC1'
        ft.bioc_document = lambda *a, **k: {'passages': [
            {'infons': {'section_type': 'RESULTS'}, 'text': 'They improved accuracy.'}]}
        try:
            result = ft.load_pmc_fulltext(Paper(source='pubmed', source_id='123', title='t'))
            self.assertTrue(result.ok)
            self.assertEqual(result.sections[0].name, 'RESULTS')
        finally:
            ft.pmc_id, ft.bioc_document = origin_id, origin_doc

    def test_load_fulltext_routes_pubmed_to_pmc(self):
        import app.fulltext as ft
        origin = ft.load_pmc_fulltext
        seen = []
        ft.load_pmc_fulltext = lambda paper, **k: (seen.append(paper.uid) or ft.FullText(source_id='x', error='stub'))
        try:
            ft.load_fulltext(Paper(source='pubmed', source_id='999', title='t'))
        finally:
            ft.load_pmc_fulltext = origin
        self.assertEqual(seen, ['pubmed:999'])


class TestBiocRetry(unittest.TestCase):
    def test_retries_transient_failure_then_succeeds(self):
        import app.fulltext as ft
        calls = {'n': 0}

        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self):
                return b'[{"documents": [{"passages": [{"infons": {}, "text": "body"}]}]}]'

        original_open, original_sleep = ft.urllib.request.urlopen, ft.time.sleep

        def flaky(*args, **kwargs):
            calls['n'] += 1
            if calls['n'] < 2:
                raise OSError('transient upstream failure')
            return FakeResponse()

        ft.urllib.request.urlopen = flaky
        ft.time.sleep = lambda seconds: None
        try:
            document = ft.bioc_document('PMC1', attempts=3)
            self.assertEqual(calls['n'], 2)
            self.assertEqual(document['passages'][0]['text'], 'body')
        finally:
            ft.urllib.request.urlopen = original_open
            ft.time.sleep = original_sleep

    def test_gives_up_after_the_last_attempt(self):
        import app.fulltext as ft
        calls = {'n': 0}
        original_open, original_sleep = ft.urllib.request.urlopen, ft.time.sleep

        def always_fail(*args, **kwargs):
            calls['n'] += 1
            raise OSError('upstream down')

        ft.urllib.request.urlopen = always_fail
        ft.time.sleep = lambda seconds: None
        try:
            with self.assertRaises(RuntimeError):
                ft.bioc_document('PMC1', attempts=3)
            self.assertEqual(calls['n'], 3)
        finally:
            ft.urllib.request.urlopen = original_open
            ft.time.sleep = original_sleep
