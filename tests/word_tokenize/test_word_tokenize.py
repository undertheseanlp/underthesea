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

    def test_word_tokenize(self):
        text = u"""Tổng thống Nga coi việc Mỹ không kích căn cứ quân sự của Syria là "sự gây hấn nhằm vào một quốc gia có chủ quyền", gây tổn hại đến quan hệ Moscow-Washington."""
        word_tokenize(text)

    def test_word_tokenize_2(self):
        import signal
        def timeout_handler(signum, frame):
            raise Exception("Timeout Exception")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(1)
        try:
            text = u"""000000000000_753889211466429	"""
            tokens = word_tokenize(text)
        except Exception as e:
            raise (e)
