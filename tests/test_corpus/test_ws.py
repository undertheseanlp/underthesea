from os.path import dirname, join
from unittest import TestCase
from underthesea.corpus.ws import WSCorpus


class TestWSCorpus(TestCase):
    def setUp(self):
        data_folder = dirname(__file__)
        self.folder = join(data_folder, "sample_ws_corpus")
        self.saved_ws_folder = join(data_folder, "sample_saved_ws_corpus")

    def test_load_column_format(self):
        corpus = WSCorpus()
        corpus.load(self.folder, "COLUMN")
        self.assertEqual(1561, len(corpus.documents[0].sentences))
        self.assertGreater(len(corpus.documents[0].sentences[0]), 0)

    def test_save_text_format(self):
        corpus = WSCorpus()
        corpus.load(self.folder, "COLUMN")
        corpus.save(self.saved_ws_folder, "TEXT")
