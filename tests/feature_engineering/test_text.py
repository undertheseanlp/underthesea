# -*- coding: utf-8 -*-

from unittest import TestCase
from underthesea.feature_engineering.text import Text, is_unicode


class TestText(TestCase):
    def test_text_1(self):
        input = u"đi học"
        output = Text(input)
        self.assertTrue(is_unicode(output))

    def test_text_2(self):
        input = "đi học"
        output = Text(input)
        self.assertTrue(is_unicode(output))

    def test_text_3(self):
        # string in unicode tổ hợp
        input = u"cộng hòa xã hội"
        output = Text(input)
        self.assertTrue(is_unicode(output))

    def test_text_4(self):
        # string in byte
        input = u"đi học".encode("utf-8")
        output = Text(input)
        self.assertTrue(is_unicode(output))
