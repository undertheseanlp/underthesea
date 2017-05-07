# -*- coding: utf-8 -*-
from unittest import TestCase

from models.crf_model.features.feature_2 import word2features, template2features

sentence_1 = [(u"Chào", "V"), (u"em", "N"), (u"cô gái", "N"), (u"Lam Hồng", "N")]


class TestDynamicFeature(TestCase):

    def test_1(self):
        feature = template2features(sentence_1, 0, "T[-1][1]")
        self.assertEqual([u"T[-1][1]=BOS"], feature)

    def test_2(self):
        feature = template2features(sentence_1, 1, "T[-1][1]")
        self.assertEqual([u"T[-1][1]=V"], feature)

    def test_3(self):
        feature = template2features(sentence_1, 2, "T[-2,-1][1]")
        self.assertEqual([u"T[-2,-1][1]=V N"], feature)
