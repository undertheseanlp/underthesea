# -*- coding: utf-8 -*-
from unittest import TestCase
from underthesea import pos_tag


class TestPostag2(TestCase):
    def test_simple_cases(self):
        sentence = u""
        actual = pos_tag(sentence)
        expected = []
        self.assertEqual(actual, expected)

    def test_accuracy(self):
        text = "Tổng Bí thư: Ai trót để tay nhúng chàm thì hãy sớm tự gột rửa"
        output = pos_tag(text, model="v2.0")
        self.assertEqual(len(output), 13)
