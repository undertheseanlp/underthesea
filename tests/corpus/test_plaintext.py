import os
import shutil
from os import listdir
from unittest import TestCase
from os.path import join
import sys
from underthesea.corpus import PlainTextCorpus

if sys.version_info >= (3, 0):
    unicode = str

test_folder = os.path.dirname(__file__)


class TestPlainTextCorpus(TestCase):
    def setUp(self):
        self.plaintext_folder = join(test_folder, "sample_text_corpus")
        self.saved_plaintext_folder = join(test_folder, "sample_saved_text_corpus")

    def test___init__(self):
        corpus = PlainTextCorpus()
        self.assertIsNone(corpus.documents)

    def test_load(self):
        corpus = PlainTextCorpus()
        corpus.load(self.plaintext_folder)
        self.assertEqual(4, len(corpus.documents))

    def test_sentences(self):
        corpus = PlainTextCorpus()
        corpus.load(self.plaintext_folder)
        self.assertEqual(list, type(corpus.documents[0].sentences))

    def test_unicode(self):
        corpus = PlainTextCorpus()
        corpus.load(self.plaintext_folder)
        self.assertEqual(unicode, type(corpus.documents[0].sentences[0]))

    def test_save(self):
        corpus = PlainTextCorpus()
        corpus.load(self.plaintext_folder)
        corpus.save(self.saved_plaintext_folder)
        files = listdir(self.saved_plaintext_folder)
        self.assertEqual(4, len(files))
        try:
            shutil.rmtree(self.saved_plaintext_folder)
        except Exception:
            pass
