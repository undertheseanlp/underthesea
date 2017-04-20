# -*- coding: utf-8 -*-
from unittest import TestCase
from models.crf_model.features.feature_2 import word2features, template2features

sentence_1 = [(u"Chào", "V"), (u"em", "N"), (u"cô gái", "N"), (u"Lam Hồng", "N")]


class TestWord2features(TestCase):
    def test_function_1(self):
        feature = template2features(sentence_1, 0, "T[0].lower")
        self.assertEqual([u"T[0].lower=chào"], feature)

    def test_function_2(self):
        feature = template2features(sentence_1, 0, "T[-1].lower")
        self.assertEqual([u"T[-1].lower=BOS"], feature)

    def test_function_3(self):
        feature = template2features(sentence_1, 1, "T[-1].lower")
        self.assertEqual([u"T[-1].lower=chào"], feature)

