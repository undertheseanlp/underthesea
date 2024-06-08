# -*- coding: utf-8 -*-
from unittest import TestCase
from underthesea import lang_detect


class TestLangDetect(TestCase):
    def test_lang_detect_1(self):
        actual = lang_detect("Bộ Công Thương xóa một tổng cục, giảm nhiều đầu mối")
        expected = "vi"
        self.assertEqual(actual, expected)

    def test_lang_detect_2(self):
        actual = lang_detect("Ceci est un texte français.")
        expected = "fr"
        self.assertEqual(actual, expected)

    def test_lang_detect_3(self):
        actual = lang_detect("如來の妙色身、 世間與に等しきは無し。")
        expected = "ja"
        self.assertEqual(actual, expected)
