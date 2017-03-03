import shutil
from os import listdir
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

    def test_save(self):
        corpus = PlainTextCorpus()
        corpus.load("sample_text_corpus")
        dist = "sample_saved_text_corpus"
        corpus.save(dist)
        files = listdir(dist)
        self.assertEqual(4, len(files))
        try:
            shutil.rmtree(dist)
        except:
            pass
