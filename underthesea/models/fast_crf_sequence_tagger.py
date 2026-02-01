from os.path import join
from pathlib import Path

import joblib
from underthesea_core import CRFFeaturizer, CRFTagger


class FastCRFSequenceTagger:
    def __init__(self, features=None, dictionary=None):
        if dictionary is None:
            dictionary = set()
        self.features = features
        self.dictionary = dictionary
        self.estimator = None
        self.featurizer = None
        self.path_model = "models.bin"
        self.path_features = "features.bin"
        self.path_dictionary = "dictionary.bin"

    def forward(self, samples, contains_labels=False):
        return self.featurizer.process(samples, contains_labels)

    def save(self, base_path):
        joblib.dump(self.features, join(base_path, self.path_features))
        joblib.dump(self.dictionary, join(base_path, self.path_dictionary))

    def load(self, base_path):
        model_path = str(Path(base_path) / self.path_model)
        estimator = CRFTagger()
        estimator.load(model_path)
        features = joblib.load(join(base_path, self.path_features))
        dictionary = joblib.load(join(base_path, self.path_dictionary))
        featurizer = CRFFeaturizer(features, dictionary)
        self.featurizer = featurizer
        self.estimator = estimator

    def predict(self, tokens):
        # Optimized: featurize + tag in single Rust call
        return self.estimator.tag_tokens(tokens, self.featurizer)

    def predict_batch(self, sequences):
        # Optimized: batch featurize + tag in Rust
        return self.estimator.tag_batch(sequences, self.featurizer)
