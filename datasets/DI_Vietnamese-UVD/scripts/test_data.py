from os.path import join, dirname
from unittest import TestCase

from data import Dictionary

CORPUS_FOLDER = dirname(dirname(__file__))
DICT_FOLDER = join(CORPUS_FOLDER, "data", "underthesea_v170")
dict_file = join(DICT_FOLDER, 'underthesea_dictionary.yaml')


class TestDictionary(TestCase):
    def test_load(self):
        Dictionary.load(dict_file)
