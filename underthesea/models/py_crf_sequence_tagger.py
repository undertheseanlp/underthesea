from os.path import join
from pathlib import Path

import joblib
import pycrfsuite
from underthesea.transformer.tagged import TaggedTransformer


class PyCRFSequenceTagger:
    def __init__(self, features=None, dictionary=set()):
        self.features = features
        self.dictionary = dictionary
        self.crf_tagger = None
        self.path_model = "models.bin"
        self.path_features = "features.bin"
        self.path_dictionary = "dictionary.bin"

    def save(self, base_path):
        joblib.dump(self.features, join(base_path, self.path_features))

    def load(self, base_path):
        model_path = str(Path(base_path) / self.path_model)
        crf_tagger = pycrfsuite.Tagger()
        crf_tagger.open(model_path)
        features = joblib.load(join(base_path, self.path_features))
        featurizer = TaggedTransformer(features)
        self.featurizer = featurizer
        self.crf_tagger = crf_tagger

    def predict(self, tokens):
        tokens = [[token] for token in tokens]
        X = self.featurizer.transform([tokens])[0]
        tags = self.crf_tagger.tag(X)
        return tags
