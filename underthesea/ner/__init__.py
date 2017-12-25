from underthesea import chunk
import sys

if sys.version_info >= (3, 0):
    from .model_crf import CRFNERPredictor
else:
    from model_crf import CRFNERPredictor


def ner(sentence, format=None):
    """
    Vietnamese Named Entity Recognition

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
    >>> from underthesea import ner
    >>> sentence = "Ông Putin ca ngợi những thành tựu vĩ đại của Liên Xô"
    >>> ner(sentence)
    [('Ông', 'Nc', 'B-NP', 'O'),
    ('Putin', 'Np', 'B-NP', 'B-PER'),
    ('ca ngợi', 'V', 'B-VP', 'O'),
    ('những', 'L', 'B-NP', 'O'),
    ('thành tựu', 'N', 'B-NP', 'O'),
    ('vĩ đại', 'A', 'B-AP', 'O'),
    ('của', 'E', 'B-PP', 'O'),
    ('Liên Xô', 'Np', 'B-NP', 'B-LOC')]
    """
    sentence = chunk(sentence)
    crf_model = CRFNERPredictor.Instance()
    result = crf_model.predict(sentence, format)
    return result
