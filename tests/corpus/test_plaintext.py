from unittest import TestCase

from underthesea.corpus import PlainTextCorpus


class TestPlainTextCorpus(TestCase):
    def test___init__(self):
        corpus = PlainTextCorpus()
        self.assertIsNone(corpus.documents)

    def test_load(self):
        corpus = PlainTextCorpus()
        corpus.load("sample_text_corpus")
        self.assertEqual(4, len(corpus.documents))
