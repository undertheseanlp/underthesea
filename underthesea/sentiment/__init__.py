# -*- coding: utf-8 -*-
from underthesea.sentiment import bank


def sentiment(X, domain=None):
    """
    Sentiment Analysis

    Parameters
    ==========

    X: {unicode, str}
        raw sentence
    domain: {None, 'bank'}

        None: general domain
        bank: bank domain
    Returns
    =======
    tokens: list
        sentiment of sentence

    Examples
    --------

    >>> # -*- coding: utf-8 -*-
    >>> from underthesea import classify
    >>> sentence = "Vừa smartbidv, vừa bidv online mà lại k dùng chung 1 tài khoản đăng nhập, rắc rối!"
    >>> classify(sentence, domain='bank')
    ('INTERNET BANKING#NEGATIVE',)
    """
    if X == "":
        return None
    if domain == 'bank':
        return bank.sentiment(X)
