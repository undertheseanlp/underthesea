# -*- coding: utf-8 -*-

from unittest import TestCase

from underthesea.dictionary import Dictionary
from underthesea.feature_engineering.text import Text


class TestDictionary(TestCase):
    def test_1(self):
        dictionary = Dictionary.Instance()
        word = dictionary.lookup(Text("đi"))
        self.assertEqual(Text("đi"), word["text"])
        self.assertEqual(Text("động từ"), word["pos"])

    def test_2(self):
        dictionary = Dictionary.Instance()
        word = dictionary.lookup(Text("không có từ này"))
        self.assertEqual(None, word)
