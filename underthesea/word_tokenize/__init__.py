# -*- coding: utf-8 -*-
from .regex_tokenize import tokenize
from .model import CRFModel


def word_tokenize(sentence, format=None):
    """
    Vietnamese word segmentation

    Parameters
    ==========

    sentence: {unicode, str}
        raw sentence

    Returns
    =======
    tokens: list of text
        tagged sentence

    Examples
    --------

    >>> # -*- coding: utf-8 -*-
    >>> from underthesea import word_tokenize
    >>> sentence = "Bác sĩ bây giờ có thể thản nhiên báo tin bệnh nhân bị ung thư"

    >>> word_tokenize(sentence)
    ['Bác sĩ', 'bây giờ', 'có thể', 'thản nhiên', 'báo tin', 'bệnh nhân', 'bị', 'ung thư']

    >>> word_tokenize(sentence, format="text")
    'Bác_sĩ bây_giờ có_thể thản_nhiên báo_tin bệnh_nhân bị ung_thư'
    """
    tokens = tokenize(sentence)
    crf_model = CRFModel.instance()
    output = crf_model.predict(tokens, format)
    tokens = [token[0] for token in output]
    tags = [token[1] for token in output]
    output = []
    for tag, token in zip(tags, tokens):
        if tag == "I-W":
            output[-1] = output[-1] + u" " + token
        else:
            output.append(token)
    if format == "text":
        output = u" ".join([item.replace(" ", "_") for item in output])
    return output
