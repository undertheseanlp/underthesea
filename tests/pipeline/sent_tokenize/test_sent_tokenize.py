# -*- coding: utf-8 -*-
from unittest import TestCase
from underthesea import sent_tokenize


class TestSentTokenize(TestCase):
    def test_simple_1(self):
        text = ""
        actual = sent_tokenize(text)
        expected = []
        self.assertEqual(actual, expected)

    def test_simple_2(self):
        text = "hôm nay"
        actual = sent_tokenize(text)
        expected = [
            "hôm nay"
        ]
        self.assertEqual(actual, expected)

    def test_1(self):
        text = "Taylor cho biết lúc đầu cô cảm thấy ngại với cô bạn thân Amanda nhưng rồi mọi thứ trôi qua nhanh chóng. Amanda cũng thoải mái với mối quan hệ này."
        actual = sent_tokenize(text)
        expected = [
            "Taylor cho biết lúc đầu cô cảm thấy ngại với cô bạn thân Amanda nhưng rồi mọi thứ trôi qua nhanh chóng.",
            "Amanda cũng thoải mái với mối quan hệ này."
        ]
        self.assertEqual(actual, expected)
