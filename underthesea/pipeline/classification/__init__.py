# -*- coding: utf-8 -*-


def classify(X, domain=None):
    """
    Text classification

    Parameters
    ==========

    X: {unicode, str}
        raw sentence
    domain: {None, 'bank'}
        domain of text
            * None: general domain
            * bank: bank domain
    Returns
    =======
    tokens: list
        categories of sentence

    """
    if X == "":
        return None
    if domain == 'bank':
        from underthesea.pipeline.classification import bank
        return bank.classify(X)
    from underthesea.pipeline.classification import vntc
    return vntc.classify(X)
