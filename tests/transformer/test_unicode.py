# -*- coding: utf-8 -*-
from unittest import TestCase
from underthesea.transformer.unicode import UnicodeTransformer
import sys
if sys.version_info >= (3, 0):
    unicode = str


class TestUnicodeTransformer(TestCase):
    def test_transform_1(self):
        transformer = UnicodeTransformer()
        s = "tôi đi học"
        self.assertEqual(unicode, type(transformer.transform(s)))

    def test_transform_2(self):
        transformer = UnicodeTransformer()
        s = u"tôi đi học"
        self.assertEqual(unicode, type(transformer.transform(s)))

