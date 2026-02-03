# -*- coding: utf-8 -*-
from unittest import TestCase
from underthesea import classify


class TestGeneralClassify(TestCase):
    def test_labels_returns_list(self):
        self.assertGreater(len(classify.labels), 0)

    def test_labels_contains_expected_labels(self):
        expected_labels = ['The thao', 'Vi tinh', 'Suc khoe', 'Kinh doanh']
        for label in expected_labels:
            self.assertIn(label, classify.labels)

    def test_classify_result_in_labels(self):
        text = "HLV ngoại đòi gần tỷ mỗi tháng dẫn dắt tuyển Việt Nam"
        result = classify(text)
        self.assertIn(result, classify.labels)

    def test_classify_null_cases(self):
        sentence = u""
        actual = classify(sentence)
        expected = None
        self.assertEqual(actual, expected)

    def test_classify_simple_case(self):
        text = u"HLV ngoại đòi gần tỷ mỗi tháng dẫn dắt tuyển Việt Nam"
        actual = classify(text)
        expected = "The thao"
        self.assertEqual(actual, expected)

    def test_classify_sports(self):
        text = u"Việt Nam giành chiến thắng 3-0 trước Thái Lan trong trận bán kết"
        actual = classify(text)
        expected = "The thao"
        self.assertEqual(actual, expected)

    def test_classify_technology(self):
        text = u"Apple ra mắt iPhone mới với nhiều tính năng đột phá"
        actual = classify(text)
        expected = "Van hoa"
        self.assertEqual(actual, expected)

    def test_classify_health(self):
        text = u"Phát hiện vaccine mới chống lại virus corona"
        actual = classify(text)
        expected = "Suc khoe"
        self.assertEqual(actual, expected)

    def test_classify_business(self):
        text = u"Thị trường chứng khoán tăng điểm mạnh trong phiên sáng nay"
        actual = classify(text)
        expected = "Kinh doanh"
        self.assertEqual(actual, expected)
