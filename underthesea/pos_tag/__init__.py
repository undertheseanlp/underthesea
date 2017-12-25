from underthesea import word_sent
from .model_crf import CRFPOSTagPredictor


def pos_tag(sentence, format=None):
    """
    Vietnamese POS tagging

    Parameters
    ==========

    sentence: {unicode, str}
        Raw sentence

    Returns
    =======
    tokens: list of text
        tagged sentence
    """
    sentence = word_sent(sentence)
    crf_model = CRFPOSTagPredictor.Instance()
    result = crf_model.predict(sentence, format)
    return result

