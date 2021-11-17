from underthesea.transformer.tagged import TaggedTransformer
from os.path import join, dirname
import pycrfsuite
from underthesea_core import CRFFeaturizer

from underthesea.transformer.tagged_feature import lower_words

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

transformer = TaggedTransformer(template)

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
