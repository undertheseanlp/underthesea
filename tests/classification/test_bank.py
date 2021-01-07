# -*- coding: utf-8 -*-
from unittest import TestCase
from underthesea import classify


class TestBankClassify(TestCase):
    def test_classify_simple_case(self):
        text = u"Mở tài khoản ATM thì có đc quà ko ad"
        actual = classify(text, domain='bank')
        expected = ["CUSTOMER_SUPPORT"]
        self.assertEqual(expected, actual)

    def test_classify_simple_case_2(self):
        text = u"Dkm t chuyển vẫn bị mất phí"
        actual = classify(text, domain='bank')
        expected = ['INTEREST_RATE', 'MONEY_TRANSFER']
        self.assertEqual(expected, actual)

    def test_classify_simple_case_3(self):
        text = u"Lãi suất từ BIDV rất ưu đãi"
        actual = classify(text, domain='bank')
        expected = ['DISCOUNT']
        self.assertEqual(expected, actual)
