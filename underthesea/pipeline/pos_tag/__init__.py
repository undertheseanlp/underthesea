# -*- coding: utf-8 -*-
from underthesea import word_tokenize
from .model_crf import CRFPOSTagPredictor

pos_model_v2 = None


def pos_tag(sentence, format=None, model=None):
    global pos_model_v2
    """
    Vietnamese POS tagging

    Parameters
    ==========

    sentence: {unicode, str}
        Raw sentence

    Returns
    =======
    tokens: list of tuple with word, pos tag
        tagged sentence
    Examples
    --------
    >>> # -*- coding: utf-8 -*-
    >>> from underthesea import pos_tag
    >>> sentence = "Chợ thịt chó nổi tiếng ở TPHCM bị truy quét"
    >>> pos_tag(sentence)
    [('Chợ', 'N'),
    ('thịt', 'N'),
    ('chó', 'N'),
    ('nổi tiếng', 'A'),
    ('ở', 'E'),
    ('TPHCM', 'Np'),
    ('bị', 'V'),
    ('truy quét', 'V')]
    """
    sentence = word_tokenize(sentence)
    if model == "v2.0":
        crf_model = CRFPOSTagPredictor.Instance(model="v2.0")
    else:
        crf_model = CRFPOSTagPredictor.Instance()
    result = crf_model.predict(sentence, format)
    return result
