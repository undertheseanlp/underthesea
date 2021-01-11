# -*- coding: utf-8 -*-
from unittest import TestCase
from underthesea import chunk


class TestChunk(TestCase):
    def test_simple_cases(self):
        sentence = u""
        actual = chunk(sentence)
        expected = []
        self.assertEqual(actual, expected)

    def test_accuracy(self):
        output = chunk(
            u"Tổng Bí thư: Ai trót để tay nhúng chàm thì hãy sớm tự gột rửa")
        self.assertEqual(len(output), 13)
