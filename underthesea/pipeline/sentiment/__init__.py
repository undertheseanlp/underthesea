# -*- coding: utf-8 -*-


def sentiment(X, domain='general'):
    """
    Sentiment Analysis

    Parameters
    ==========

    X: str
        raw sentence
    domain: str
        domain of text (bank or general). Default: `general`

    Returns
    =======
        Text: Text of input sentence
        Labels: Sentiment of sentence

    Examples
    --------

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
