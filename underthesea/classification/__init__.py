# -*- coding: utf-8 -*-
from underthesea.classification import bank


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
        return bank.classify(X)

