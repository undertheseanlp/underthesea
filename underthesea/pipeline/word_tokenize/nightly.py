# -*- coding: utf-8 -*-
from underthesea_core import CRFFeaturizer

from .regex_tokenize import tokenize
from os.path import join, dirname
import pycrfsuite

from ...transformer.tagged_feature import lower_words

template = [
    "T[-2].lower", "T[-1].lower", "T[0].lower", "T[1].lower", "T[2].lower",

    "T[-1].isdigit", "T[0].isdigit", "T[1].isdigit",

    "T[-1].istitle", "T[0].istitle", "T[1].istitle",
    "T[0,1].istitle", "T[0,2].istitle",

    "T[-2].is_in_dict", "T[-1].is_in_dict", "T[0].is_in_dict", "T[1].is_in_dict", "T[2].is_in_dict",
    "T[-2,-1].is_in_dict", "T[-1,0].is_in_dict", "T[0,1].is_in_dict", "T[1,2].is_in_dict",
    "T[-2,0].is_in_dict", "T[-1,1].is_in_dict", "T[0,2].is_in_dict",

    # word unigram and bigram and trigram
    "T[-2]", "T[-1]", "T[0]", "T[1]", "T[2]",
    "T[-2,-1]", "T[-1,0]", "T[0,1]", "T[1,2]",
    "T[-2,0]", "T[-1,1]", "T[0,2]",
]
crf_featurizer = CRFFeaturizer(template, lower_words)


class CRFModel:
    objects = {}

    def __init__(self, model_path=None):
        if not model_path:
            model_path = join(dirname(__file__), "wt_crf_2018_09_13.bin")
        estimator = pycrfsuite.Tagger()
        estimator.open(model_path)
        self.estimator = estimator

    @classmethod
    def instance(cls, model_path=None):
        if model_path not in cls.objects:
            cls.objects[model_path] = cls(model_path)
        object = cls.objects[model_path]
        return object

    def predict(self, sentence, format=None):
        tokens = [(token, "X") for token in sentence]
        x = crf_featurizer.process([tokens])[0]
        tags = self.estimator.tag(x)
        return list(zip(sentence, tags))


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
