# -*- coding: utf-8 -*-

from unittest import TestCase

from underthesea.dictionary import Dictionary
from underthesea.feature_engineering.text import Text


class TestDictionary(TestCase):
    def test_1(self):
        dictionary = Dictionary.Instance()
        senses = dictionary.lookup(Text("đi"))
        self.assertEqual(22, len(senses))
        sense = senses[0]
        self.assertEqual("V", sense["pos"])
        self.assertGreater(len(sense["definition"]), 0)

    def test_2(self):
        dictionary = Dictionary.Instance()
        word = dictionary.lookup(Text("không có từ này"))
        self.assertEqual(None, word)
