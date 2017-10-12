# -*- coding: utf-8 -*-
from unittest import TestCase
from underthesea import ner


class TestNER(TestCase):
    def test_simple_cases(self):
        sentence = u""
        actual = ner(sentence)
        expected = []
        self.assertEqual(actual, expected)

    def test_accuracy(self):
        output = ner(u"Bộ Công Thương xóa một tổng cục, giảm nhiều đầu mối")
        self.assertEqual(output[0][3], "B-ORG")
        self.assertEqual(output[1][3], "I-ORG")
