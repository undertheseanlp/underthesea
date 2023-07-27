# -*- coding: utf-8 -*-


def classify(X, domain=None, model=None):
    """
    Text classification

    Args:
        X (str): The raw sentence
        domain (str, optional): The domain of the text. Defaults to None.
            Options include:
                - None: general domain
                - 'bank': bank domain
        model (str, optional): The classification model. Defaults to None.
            Options include:
                - None: default underthesea classifier
                - 'prompt': OpenAI prompt model

    Returns:
        list: A list containing the categories of the sentence
    """
    if X == "":
        return None

    if model == 'prompt':
        from underthesea.pipeline.classification import classification_prompt
        args = {
            "domain": domain
        }
        return classification_prompt.classify(X, **args)

    if domain == 'bank':
        from underthesea.pipeline.classification import bank
        return bank.classify(X)

    from underthesea.pipeline.classification import vntc
    return vntc.classify(X)
