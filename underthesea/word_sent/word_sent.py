# -*- coding: utf-8 -*-
from underthesea.word_sent.model import CRFModel
from tokenize import tokenize


def word_sent(sentence, format=None):
    """
    :param unicode|str sentence: raw sentence
    :return: segmented sentence
    :rtype: unicode|str
    """
    sentence = tokenize(sentence)
    crf_model = CRFModel.Instance()
    result = crf_model.predict(sentence, format)
    return result


