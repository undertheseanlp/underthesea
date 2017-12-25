# -*- coding: utf-8 -*-
from unittest import TestCase
from underthesea.sentiment import sentiment


class TestSentiment(TestCase):
    def test_sentiment(self):
        text = ""
        actual = sentiment(text)
        expected = None
        self.assertEqual(actual, expected)

    def test_sentiment_1(self):
        text = "Gọi mấy lần mà lúc nào cũng là các chuyên viên đang bận hết ạ"
        actual = sentiment(text, domain="bank")
        expected = ('CUSTOMER SUPPORT#NEGATIVE',)
        self.assertEqual(actual, expected)

    def test_sentiment_2(self):
        text = "bidv cho vay hay ko phu thuoc y thich cua thang tham dinh, ko co quy dinh ro rang"
        actual = sentiment(text, domain="bank")
        expected = ('LOAN#NEGATIVE',)
        self.assertEqual(actual, expected)

    def test_sentiment_3(self):
        text = "Vừa smartbidv, vừa bidv online mà lại k dùng chung 1 tài khoản đăng nhập, rắc rối!"
        actual = sentiment(text, domain="bank")
        expected = ('INTERNET BANKING#NEGATIVE',)
        self.assertEqual(actual, expected)

    def test_sentiment_4(self):
        text = "Không tin tưởng vào ngân hàng BIDV"
        actual = sentiment(text, domain="bank")
        expected = ('TRADEMARK#NEGATIVE',)
        self.assertEqual(actual, expected)

    def test_sentiment_5(self):
        text = "Chương trình này của BIDV thật ý nghĩa"
        actual = sentiment(text, domain="bank")
        expected = ('PROMOTION#POSITIVE',)
        self.assertEqual(actual, expected)

    def test_sentiment_6(self):
        text = "mình cũng vui vì tiết kệm được thời gian"
        actual = sentiment(text, domain="bank")
        expected = ('PAYMENT#POSITIVE',)
        self.assertEqual(actual, expected)
