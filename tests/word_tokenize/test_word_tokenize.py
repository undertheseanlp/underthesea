# -*- coding: utf-8 -*-
from unittest import TestCase
from underthesea import word_tokenize


class TestWordTokenize(TestCase):
    def test_simple_cases(self):
        sentence = u""
        actual = word_tokenize(sentence)
        expected = []
        self.assertEqual(actual, expected)

        actual = word_tokenize(sentence, format="text")
        expected = u""
        self.assertEqual(actual, expected)

    def test_special_cases_2(self):
        sentence = u"="
        actual = word_tokenize(sentence)
        expected = ["="]
        self.assertEqual(actual, expected)

    def test_special_cases_3(self):
        sentence = u"=))"
        actual = word_tokenize(sentence)
        expected = ["=))"]
        self.assertEqual(actual, expected)

    def test_decomposed_from(self):
        text = u"yếu"
        acutal = word_tokenize(text)
        expected = [u'yếu']
        self.assertEqual(acutal, expected)

    def test_wordsent(self):
        text = u"""Tổng thống Nga coi việc Mỹ không kích căn cứ quân sự của Syria là "sự gây hấn nhằm vào một quốc gia có chủ quyền", gây tổn hại đến quan hệ Moscow-Washington."""
        word_tokenize(text)



