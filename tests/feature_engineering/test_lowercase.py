# -*- coding: utf-8 -*-
from unittest import TestCase

from underthesea.feature_engineering.lowercase import LowercaseTransformer


class TestUnicodeTransformer(TestCase):
    def test_transform_1(self):
        transformer = LowercaseTransformer()
        s = u"ĐI HỌC"
        output = transformer.transform(s)
        self.assertEqual(u"đi học", output)
