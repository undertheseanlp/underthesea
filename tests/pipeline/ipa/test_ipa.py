# -*- coding: utf-8 -*-
from unittest import TestCase
from underthesea.pipeline.ipa import vietnamese_to_ipa


class TestIPA(TestCase):
    def test_1(self):
        actual = vietnamese_to_ipa("trồng")
        expected = "tɕoŋ³²"
        self.assertEqual(expected, actual)
