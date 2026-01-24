# -*- coding: utf-8 -*-
from unittest import TestCase
from underthesea.pipeline.sentiment import sentiment


class TestSentiment(TestCase):
    def test_labels_returns_list(self):
        self.assertEqual(len(sentiment.labels), 2)

    def test_labels_contains_expected_labels(self):
        self.assertIn('positive', sentiment.labels)
        self.assertIn('negative', sentiment.labels)

    def test_no_text(self):
        text = ""
        actual = sentiment(text)
        expected = None
        self.assertEqual(expected, actual)

    def test_1(self):
        text = "hàng kém chất lg,chăn đắp lên dính lông lá khắp người. thất vọng"
        actual = sentiment(text, domain="general")
        expected = "negative"
        self.assertEqual(expected, actual)

    def test_2(self):
        text = "Chất lượng tốt, đóng gói cẩn thận."
        actual = sentiment(text, domain="general")
        expected = "positive"
        self.assertEqual(expected, actual)
