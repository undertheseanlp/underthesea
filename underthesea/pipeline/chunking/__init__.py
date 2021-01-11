# -*- coding: utf-8 -*-
from underthesea import pos_tag
import sys

if sys.version_info >= (3, 0):
    from .model_crf import CRFChunkingPredictor
else:
    from model_crf import CRFChunkingPredictor


def chunk(sentence, format=None):
    """
    Vietnamese chunking

    Parameters
    ==========

    sentence: {unicode, str}
        raw sentence

    Returns
    =======
    tokens: 	list of tuple with word, pos tag, chunking tag
        tagged sentence

    Examples
    --------

    >>> # -*- coding: utf-8 -*-
    >>> from underthesea import chunk
    >>> sentence = "Nghi vấn 4 thi thể Triều Tiên trôi dạt bờ biển Nhật Bản"
    >>> chunk(sentence)
    [('Nghi vấn', 'N', 'B-NP'),
    ('4', 'M', 'B-NP'),
    ('thi thể', 'N', 'B-NP'),
    ('Triều Tiên', 'Np', 'B-NP'),
    ('trôi dạt', 'V', 'B-VP'),
    ('bờ biển', 'N', 'B-NP'),
    ('Nhật Bản', 'Np', 'B-NP')]
    """
    sentence = pos_tag(sentence)
    crf_model = CRFChunkingPredictor.Instance()
    result = crf_model.predict(sentence, format)
    return result
