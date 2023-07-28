# -*- coding: utf-8 -*-
from unittest import TestCase
from underthesea import classify


class TestPromptClassify(TestCase):
    def test_prompt_1(self):
        text = "HLV ngoại đòi gần tỷ mỗi tháng dẫn dắt tuyển Việt Nam"
        actual = classify(text, model='prompt')
        expected = "Thể thao"
        self.assertEqual(actual, expected)
