import logging
import os
import shutil
from os.path import dirname, join
from pathlib import Path

import joblib
import pycrfsuite
from seqeval.metrics import classification_report
from underthesea_core import CRFFeaturizer

from underthesea.transformer.tagged_feature import lower_words as dictionary


class FastCRFSequenceTagger:
    def __init__(self, features=[], dictionary=set()):
        self.features = features
        self.dictionary = dictionary
        self.crf_tagger = None
        self.featurizer = CRFFeaturizer(self.features, self.dictionary)
        self.path_model = "models.bin"
        self.path_features = "features.bin"
        self.path_dictionary = "dictionary.bin"

    def forward(self, samples, contains_labels=False):
        return self.featurizer.transform(samples, contains_labels)

    def save(self, base_path):
        print("save features")
        joblib.dump(self.features, join(base_path, self.path_features))
        joblib.dump(self.dictionary, join(base_path, self.path_dictionary))

    def load(self, base_path):
        print(base_path)
        model_path = str(Path(base_path) / self.path_model)
        crf_tagger = pycrfsuite.Tagger()
        crf_tagger.open(model_path)
        features = joblib.load(join(base_path, self.path_features))
        dictionary = joblib.load(join(base_path, self.path_dictionary))
        featurizer = CRFFeaturizer(features, dictionary)
        self.featurizer = featurizer
        self.crf_tagger = crf_tagger

    def predict(self, tokens):
        tokens = [[token] for token in tokens]
        try:
            x = self.featurizer.process(tokens)
        except Exception as e:
            print(e)
        # print(x)
        # print(len(x)

        # print(len(x[0]))
        tags = [self.crf_tagger.tag(item)[0] for item in x]
        return tags