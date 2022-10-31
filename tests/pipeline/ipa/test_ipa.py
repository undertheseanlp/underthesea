# -*- coding: utf-8 -*-
from os.path import dirname, join
from unittest import TestCase
from underthesea.pipeline.ipa import viet2ipa


class TestIPA(TestCase):
    def test_1(self):
        actual = viet2ipa("trồng")
        expected = "tɕoŋ³²"
        self.assertEqual(expected, actual)

    def test_2(self):
        # text = "cún"
        text = "chật"
        actual = viet2ipa(text)
        # expected = "kun³⁴"
        expected = "ʨɤ̆t¹⁰ˀ"
        self.assertEqual(expected, actual)

    def test_3(self):
        text = 'hai âm tiết'
        actual = viet2ipa(text)
        expected = ""
        self.assertEqual(expected, actual)

    def test_4(self):
        cw = dirname(__file__)
        with open(join(cw, 'tests.txt')) as f:
            lines = f.readlines()
        items = [line.strip() for line in lines]
        for item in items:
            syllable, ipa = item.split(",")
            actual = viet2ipa(syllable)
            self.assertEqual(ipa, actual)
