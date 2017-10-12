from underthesea import chunk


def ner(text, format=None):
    """
    location and classify named entities in text

    :param text: raw text
    :param format:
    :return: list
    """
    text = chunk(text)
    # model = NERCRFModel.Instance()
    model = None
    result = model.predict(text, format)
    return result
