# -*- coding: utf-8 -*-
from unittest import TestCase
from underthesea.pipeline.ipa import viet2ipa


class TestIPA(TestCase):
    def test_1(self):
        actual = viet2ipa("trồng")
        expected = "tɕoŋ³²"
        self.assertEqual(expected, actual)
