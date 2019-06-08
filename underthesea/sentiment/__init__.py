# -*- coding: utf-8 -*-
from underthesea.sentiment import bank


def sentiment(X, domain=None):
    """
    Sentiment Analysis

    Parameters
    ==========

    X: {unicode, str}
        raw sentence
    domain: {'bank'}
        domain of text
            * bank: bank domain
    Returns
    =======
    List of labels that be sentiment of text

    Examples
    --------

    >>> # -*- coding: utf-8 -*-
    >>> from underthesea import sentiment
    >>> sentence = "Chuyen tiền k nhận Dc tiên"
    >>> sentiment(sentence, domain='bank')

    [MONEY_TRANSFER#negative (1.0)]
    """
    if X == "":
        return None
    if domain == 'bank':
        return bank.sentiment(X)
