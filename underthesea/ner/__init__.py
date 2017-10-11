from underthesea import chunk
from underthesea.ner.ner_crf import NERCRFModel


def ner(text, format=None):
    """
    location and classify named entities in text

    :param text: raw text
    :param format:
    :return: list
    """
    text = chunk(text)
    model = NERCRFModel.Instance()
    result = model.predict(text, format)
    return result
