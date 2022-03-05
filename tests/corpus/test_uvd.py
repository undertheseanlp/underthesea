from unittest import TestCase

from underthesea.corpus.uvd import UVD, DictionaryStat


class TestUVD(TestCase):

    def test_1(self):
        dictionary = UVD()
        words = dictionary.words
        self.assertIn('bò', words)

    def test_stat(self):
        dictionary = UVD()
        DictionaryStat.stat(dictionary)
