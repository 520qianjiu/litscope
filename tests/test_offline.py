import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.dedupe import dedupe, norm_title
from app.models import Paper
from app.sources.arxiv import ArxivSource
from app.sources.crossref import CrossrefSource
from app.sources.pubmed import PubmedSource

ARXIV_XML = (
    '<feed xmlns=\'http://www.w3.org/2005/Atom\' xmlns:arxiv=\'http://arxiv.org/schemas/atom\'>'
    '<entry>'
    '<id>http://arxiv.org/abs/2401.00001v1</id>'
    '<title>Attention   Is All You Need Again</title>'
    '<summary>We revisit attention.</summary>'
    '<published>2024-01-01T00:00:00Z</published>'
    '<author><name>Ada Lovelace</name></author>'
    '<author><name>Alan Turing</name></author>'
    '<arxiv:doi>10.1000/abc</arxiv:doi>'
    '</entry>'
    '</feed>'
)

CROSSREF_JSON = json.dumps({
    'message': {
        'items': [{
            'DOI': '10.1000/abc',
            'title': ['Attention Is All You Need Again'],
            'abstract': '<jats:p>We revisit attention.</jats:p>',
            'author': [{'given': 'Ada', 'family': 'Lovelace'}],
            'issued': {'date-parts': [[2024, 1, 1]]},
            'container-title': ['Journal of Tests'],
            'is-referenced-by-count': 42,
            'URL': 'https://doi.org/10.1000/abc',
        }]
    }
})

PUBMED_XML = (
    '<PubmedArticleSet><PubmedArticle><MedlineCitation>'
    '<PMID>12345678</PMID>'
    '<Article>'
    '<Journal><Title>Journal of Tests</Title>'
    '<JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue></Journal>'
    '<ArticleTitle>Attention Is All You Need Again.</ArticleTitle>'
    '<Abstract><AbstractText Label=\'BACKGROUND\'>We revisit attention.</AbstractText></Abstract>'
    '<AuthorList><Author><LastName>Lovelace</LastName><ForeName>Ada</ForeName></Author></AuthorList>'
    '</Article>'
    '</MedlineCitation>'
    '<PubmedData><ArticleIdList><ArticleId IdType=\'doi\'>10.1000/abc</ArticleId></ArticleIdList></PubmedData>'
    '</PubmedArticle></PubmedArticleSet>'
)


class TestNormTitle(unittest.TestCase):
    def test_ignores_case_and_punctuation(self):
        self.assertEqual(norm_title('Attention, Is All You Need!'),
                         norm_title('attention is all you need'))


class TestDedupe(unittest.TestCase):
    def test_merges_same_doi_across_sources(self):
        first = Paper(source='arxiv', source_id='2401.00001v1',
                      title='Attention Is All You Need Again',
                      abstract='from arxiv', doi='10.1000/abc', year=2024)
        second = Paper(source='crossref', source_id='10.1000/abc',
                       title='Attention Is All You Need Again',
                       doi='10.1000/ABC', citations=42, venue='Journal of Tests')
        merged = dedupe([first, second])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].citations, 42)
        self.assertEqual(merged[0].venue, 'Journal of Tests')
        self.assertEqual(merged[0].abstract, 'from arxiv')
        self.assertEqual(merged[0].merged_sources, ['crossref:10.1000/abc'])

    def test_merges_near_identical_titles(self):
        first = Paper(source='arxiv', source_id='1', title='Deep Learning for Protein Folding')
        second = Paper(source='pubmed', source_id='2', title='Deep learning for protein-folding.')
        self.assertEqual(len(dedupe([first, second])), 1)

    def test_keeps_unrelated_papers(self):
        first = Paper(source='arxiv', source_id='1', title='Alpha One')
        second = Paper(source='arxiv', source_id='2', title='Beta Two')
        self.assertEqual(len(dedupe([first, second])), 2)

    def test_drops_blank_titles(self):
        self.assertEqual(dedupe([Paper(source='arxiv', source_id='1', title='   ')]), [])


class TestArxivParse(unittest.TestCase):
    def test_fields(self):
        papers = ArxivSource().parse(ARXIV_XML)
        self.assertEqual(len(papers), 1)
        paper = papers[0]
        self.assertEqual(paper.source_id, '2401.00001v1')
        self.assertEqual(paper.title, 'Attention Is All You Need Again')
        self.assertEqual(paper.authors, ['Ada Lovelace', 'Alan Turing'])
        self.assertEqual(paper.year, 2024)
        self.assertEqual(paper.doi, '10.1000/abc')


class TestCrossrefParse(unittest.TestCase):
    def test_fields(self):
        papers = CrossrefSource().parse(CROSSREF_JSON)
        self.assertEqual(len(papers), 1)
        paper = papers[0]
        self.assertEqual(paper.doi, '10.1000/abc')
        self.assertEqual(paper.abstract, 'We revisit attention.')
        self.assertEqual(paper.citations, 42)
        self.assertEqual(paper.year, 2024)


class TestPubmedParse(unittest.TestCase):
    def test_fields(self):
        papers = PubmedSource().parse(PUBMED_XML)
        self.assertEqual(len(papers), 1)
        paper = papers[0]
        self.assertEqual(paper.source_id, '12345678')
        self.assertEqual(paper.abstract, 'BACKGROUND: We revisit attention.')
        self.assertEqual(paper.authors, ['Ada Lovelace'])
        self.assertEqual(paper.year, 2024)
        self.assertEqual(paper.doi, '10.1000/abc')
        self.assertEqual(paper.url, 'https://pubmed.ncbi.nlm.nih.gov/12345678/')


if __name__ == '__main__':
    unittest.main()
