# -*- coding: utf-8 -*-
from unittest import TestCase
# from underthesea.corpus.word_sent import word_sent
from underthesea.corpus import word_sent


class TestWord_sent(TestCase):
    def test_word_sent_1(self):
        sentence = u"cộng hòa xã hội chủ nghĩa"
        predict_sentence = word_sent(sentence)
        actual_sentence = u'cộng hòa xã_hội_chủ_nghĩa'
        self.assertEqual(actual_sentence, predict_sentence)

    def test_word_sent_2(self):
        sentence = u"nhật tinh anh đang làm gì đấy ?"
        predict_sentence = word_sent(sentence)
        actual_sentence = u'nhật tinh_anh đang làm gì đấy ?'
        self.assertEqual(actual_sentence, predict_sentence)

    def test_word_sent_3(self):
        sentence = u"hươu rất sợ tiếng động lạ ? ?"
        predict_sentence = word_sent(sentence)
        actual_sentence = u'hươu rất sợ tiếng_động lạ ? ?'
        self.assertEqual(actual_sentence, predict_sentence)

    def test_word_sent_4(self):
        sentence = u"Hồ Chí Minh ?"
        predict_sentence = word_sent(sentence)
        actual_sentence = u'Hồ_Chí_Minh ?'
        self.assertEqual(actual_sentence, predict_sentence)
