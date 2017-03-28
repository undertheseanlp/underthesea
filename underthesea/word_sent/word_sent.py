# -*- coding: utf-8 -*-
from underthesea.word_sent.model import CRFModel


def word_sent(sentence):
    """
    :param unicode|str sentence: raw sentence
    :return: segmented sentence
    :rtype: unicode|str
    """
    crf_model = CRFModel()
    return crf_model.predict(sentence)
