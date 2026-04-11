# -*- coding: utf-8 -*-
from unittest import TestCase

from apps.directory.col_data import VietnameseWiktionary


class TestColData(TestCase):
    def test_1(self):
        word = "a"
        VietnameseWiktionary.get(word)
