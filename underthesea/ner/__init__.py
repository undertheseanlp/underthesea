from underthesea import chunk
import sys

if sys.version_info >= (3, 0):
    from .model_crf import CRFNERPredictor
else:
    from model_crf import CRFNERPredictor


def ner(sentence, format=None):
    """
    part of speech tagging

    :param unicode|str sentence: raw sentence
    :return: ner tagged sentence
    :rtype: list
    """
    sentence = chunk(sentence)
    crf_model = CRFNERPredictor.Instance()
    result = crf_model.predict(sentence, format)
    return result
