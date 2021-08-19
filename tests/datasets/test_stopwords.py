# -*- coding: utf-8 -*-

from unittest import TestCase

from underthesea.datasets.stopwords import stopwords


class TestDictionary(TestCase):
    def test_1(self):
        self.assertTrue("đã" in stopwords)
