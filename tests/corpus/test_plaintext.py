from unittest import TestCase

from underthesea.corpus import PlainTextCorpus


class TestPlainTextCorpus(TestCase):
    def test___init__(self):
        corpus = PlainTextCorpus()
        self.assertIsNone(corpus.documents)

