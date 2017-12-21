# -*- coding: utf-8 -*-
from unittest import TestCase
from underthesea import classify


class TestClassify(TestCase):
    def test_classify_simple_case(self):
        text = u"Mở tài khoản ATM thì có đc quà ko ad"
        actual = classify(text, domain='bank')
        expected = ("ACCOUNT",)
        self.assertEqual(actual, expected)

    def test_classify_simple_case_2(self):
        text = u"Tôi rất thích cách phục vụ của nhân viên BIDV"
        actual = classify(text, domain='bank')
        expected = ('CUSTOMER SUPPORT',)
        self.assertEqual(actual, expected)

    def test_classify_simple_case_3(self):
        text = u"Lãi suất từ BIDV rất ưu đãi"
        actual = classify(text, domain='bank')
        expected = ('INTEREST RATE',)
        self.assertEqual(actual, expected)
