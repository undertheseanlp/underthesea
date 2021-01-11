# -*- coding: utf-8 -*-


def sentiment(X, domain='general'):
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
    >>> sentence = "Chuyen tiền k nhận Dc tiên"
    >>> sentiment(sentence, domain='bank')

    [MONEY_TRANSFER#negative (1.0)]
    """
    if X == "":
        return None
    if domain == 'general':
        from underthesea.pipeline.sentiment.general import sentiment
        return sentiment(X)
    if domain == 'bank':
        from underthesea.pipeline.sentiment.bank import sentiment
        return sentiment(X)
