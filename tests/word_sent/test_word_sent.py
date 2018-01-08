# -*- coding: utf-8 -*-
from unittest import TestCase
from underthesea import word_sent


class TestWord_sent(TestCase):
    def test_simple_cases(self):
        sentence = u""
        actual = word_sent(sentence)
        expected = []
        self.assertEqual(actual, expected)

        actual = word_sent(sentence, format="text")
        expected = u""
        self.assertEqual(actual, expected)

    def test_special_cases_2(self):
        sentence = u"="
        actual = word_sent(sentence)
        expected = ["="]
        self.assertEqual(actual, expected)

    def test_special_cases_3(self):
        sentence = u"=))"
        actual = word_sent(sentence)
        expected = ["=))"]
        self.assertEqual(actual, expected)

    def test_decomposed_from(self):
        text = u"yếu"
        acutal = word_sent(text)
        expected = [u'yếu']
        self.assertEqual(acutal, expected)

    def test_wordsent(self):
        text = u"""Tổng thống Nga coi việc Mỹ không kích căn cứ quân sự của Syria là "sự gây hấn nhằm vào một quốc gia có chủ quyền", gây tổn hại đến quan hệ Moscow-Washington."""
        word_sent(text)



