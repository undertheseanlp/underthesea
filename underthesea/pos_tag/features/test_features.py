# -*- coding: utf-8 -*-
from unittest import TestCase

from models.crf_model.features.feature_2 import word2features, template2features

sentence_1 = [(u"Chào", "V"), (u"em", "N"), (u"cô gái", "N"), (u"Lam Hồng", "N")]


class TestWord2features(TestCase):
    # def test_word2features(self):
    #     features = word2features(sentence_1, 0)

    def test_template2features(self):
        feature = template2features(sentence_1, 0, "T[0]")
        self.assertEqual([u"T[0]=Chào"], feature)

    def test_template2features_1(self):
        feature = template2features(sentence_1, 1, "T[-1]")
        self.assertEqual([u"T[-1]=Chào"], feature)

    def test_template2features_2(self):
        feature = template2features(sentence_1, 1, "T[1]")
        self.assertEqual([u"T[1]=cô gái"], feature)

    def test_template2features_3(self):
        feature = template2features(sentence_1, 0, "T[-1]")
        self.assertEqual(["T[-1]=BOS"], feature)

    def test_template2features_4(self):
        feature = template2features(sentence_1, 3, "T[1]")
        self.assertEqual(["T[1]=EOS"], feature)

    def test_template2features_4(self):
        feature = template2features(sentence_1, 3, "T[1]")
        self.assertEqual(["T[1]=EOS"], feature)

    def test_template2features_bigram_1(self):
        feature = template2features(sentence_1, 0, "T[0,1]")
        self.assertEqual([u"T[0,1]=Chào em"], feature)

    def test_template2features_bigram_2(self):
        feature = template2features(sentence_1, 0, "T[0,2]")
        self.assertEqual([u"T[0,2]=Chào em cô gái"], feature)

    def test_template2features_bigram_3(self):
        feature = template2features(sentence_1, 1, "T[-1,0]")
        self.assertEqual([u"T[-1,0]=Chào em"], feature)

    def test_template2features_bigram_4(self):
        feature = template2features(sentence_1, 0, "T[-1,2]")
        self.assertEqual([u"T[-1,2]=BOS"], feature)

    def test_template2features_bigram_5(self):
        feature = template2features(sentence_1, 0, "T[3,4]")
        self.assertEqual([u"T[3,4]=EOS"], feature)
