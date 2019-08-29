# -*- coding: utf-8 -*-
from unittest import TestCase
from underthesea.sentiment import sentiment


class TestSentiment(TestCase):
    def test_no_text(self):
        text = ""
        actual = sentiment(text)
        expected = None
        self.assertEqual(actual, expected)

    def test_1(self):
        text = "hàng kém chất lg,chăn đắp lên dính lông lá khắp người. thất vọng"
        actual = sentiment(text, domain="general")
        expected = "negative"
        self.assertEqual(actual, expected)

    def test_2(self):
        text = "Sản phẩm hơi nhỏ so với tưởng tượng nhưng chất lượng tốt, đóng gói cẩn thận."
        actual = sentiment(text, domain="general")
        expected = "positive"
        self.assertEqual(actual, expected)
