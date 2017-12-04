# -*- coding: utf-8 -*-
from unittest import TestCase
from underthesea import classify
from underthesea.feature_engineering.text import Text


class TestClassify(TestCase):
    def test_classify_simple_case(self):
        text = "Mở tài khoản ATM thì có đc quà ko ad"
        actual = classify(text, domain='bank')[0]
        expected = Text("ACCOUNT")
        self.assertEqual(actual, expected)
