# -*- coding: utf-8 -*-
from unittest import TestCase
from underthesea import classify

from underthesea.feature_engineering.text import Text


class TestClassify(TestCase):
    def test_classify_null_cases(self):
        sentence = u""
        actual = classify(sentence)
        expected = None
        self.assertEqual(actual, expected)

    def test_classify_simple_case(self):
        text = u"HLV ngoại đòi gần tỷ mỗi tháng dẫn dắt tuyển Việt Nam"
        actual = classify(text)[0]
        expected = Text("The thao")
        self.assertEqual(actual, expected)
