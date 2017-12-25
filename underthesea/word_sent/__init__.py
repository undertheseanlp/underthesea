# -*- coding: utf-8 -*-
from .regex_tokenize import tokenize
from .model_crf import CRFModel


def word_sent(sentence, format=None):
    """
    word segmentation

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
    >>> from underthesea import word_sent
    >>> sentence = u"Chúng ta thường nói đến Rau sạch, Rau an toàn để phân biệt với các rau bình thường bán ngoài chợ."

    >>> word_sent(sentence)
    [u"Chúng ta", u"thường", u"nói", u"đến", u"Rau sạch", u",", u"Rau", u"an toàn", u"để", u"phân biệt", u"với",
    u"các", u"rau", u"bình thường", u"bán", u"ngoài", u"chợ", u"."]

    >>> word_sent(sentence, format="text")
    u'Chúng_ta thường nói đến Rau_sạch , Rau an_toàn để phân_biệt với các rau bình_thường bán ngoài chợ .'
    """
    sentence = tokenize(sentence).split()
    crf_model = CRFModel.Instance()
    output = crf_model.predict(sentence, format)
    tokens = [token[0] for token in output]
    tags = [token[1] for token in output]
    output = []
    for tag, token in zip(tags, tokens):
        if tag == "IW":
            output[-1] = output[-1] + u" " + token
        else:
            output.append(token)
    if format == "text":
        output = u" ".join([item.replace(" ", "_") for item in output])
    return output
