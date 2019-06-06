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
    Text: Text of input sentence
    Labels: Sentiment of sentence

    Examples
    --------

    >>> # -*- coding: utf-8 -*-
    >>> from underthesea import sentiment
    >>> sentence = "Vừa smartbidv, vừa bidv online mà lại k dùng chung 1 tài khoản đăng nhập, rắc rối!"
    >>> sentiment(sentence, domain='bank')

    Text: Vừa smartbidv, vừa bidv online mà lại k dùng chung 1 tài khoản đăng nhập, rắc rối!
    Labels: INTERNET_BANKING#negative
    """
    if X == "":
        return None
    if domain == 'bank':
        return bank.sentiment(X)
