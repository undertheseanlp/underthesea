# -*- coding: utf-8 -*-
from underthesea.word_sent.model import CRFModel
from underthesea.word_sent.tokenize import tokenize


def word_sent(text, format=None):
    """
    :param unicode|str sentence: raw sentence
    :return: segmented sentence
    :rtype: unicode|str
    """
    text = tokenize(text)
    crf_model = CRFModel.Instance()
    output = crf_model.predict(text, format)
    return output


