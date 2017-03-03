from unittest import TestCase

from underthesea.corpus.corpus import Corpus


class TestPlainTextCorpus(TestCase):
    def test___init__(self):
        corpus = Corpus()

