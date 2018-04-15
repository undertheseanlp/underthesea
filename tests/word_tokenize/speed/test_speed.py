# -*- coding: utf-8 -*-
from unittest import TestCase
from underthesea import word_tokenize


class TestWordTokenize(TestCase):
    def test_speed(self):
        f = open("content_1k.txt")
        for line in f:
            print(line)
            word_tokenize(line)


