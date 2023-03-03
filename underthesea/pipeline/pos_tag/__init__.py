# -*- coding: utf-8 -*-
from underthesea import word_tokenize
from underthesea.models.fast_crf_sequence_tagger import FastCRFSequenceTagger
from .model_crf import CRFPOSTagPredictor
from os.path import dirname, join

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
        if pos_model_v2 is None:
            pos_model_v2 = FastCRFSequenceTagger()
            wd = dirname(__file__)
            pos_model_v2.load(join(wd, "models", "pos_crf_vlsp2013_20230303"))
            tokens = sentence
            features = [[token] for token in sentence]
            tags = pos_model_v2.predict(features)
            # output of pos_model_v2 in in BOI format B-N, B-CH, B-V,...
            # remove prefix B-
            tags = [tag[2:] for tag in tags]
            result = list(zip(tokens, tags))
    else:
        crf_model = CRFPOSTagPredictor.Instance()
        result = crf_model.predict(sentence, format)
    return result
