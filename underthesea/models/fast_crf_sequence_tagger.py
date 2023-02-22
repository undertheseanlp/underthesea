from os.path import join
from pathlib import Path
import joblib
import pycrfsuite
from underthesea_core import CRFFeaturizer


class FastCRFSequenceTagger:
    def __init__(self, features=None):
        self.features = features
        self.estimator = None
        self.featurizer = None
        self.path_model = "models.bin"
        self.path_features = "features.bin"
        self.path_dictionary = "dictionary.bin"

    def forward(self, samples, contains_labels=False):
        return self.featurizer.process(samples, contains_labels)

    def save(self, base_path):
        print("save features")
        joblib.dump(self.features, join(base_path, self.path_features))

    def load(self, base_path):
        print(base_path)
        model_path = str(Path(base_path) / self.path_model)
        estimator = pycrfsuite.Tagger()
        estimator.open(model_path)
        features = joblib.load(join(base_path, "features.bin"))
        featurizer = CRFFeaturizer(features, set())
        self.featurizer = featurizer
        self.estimator = estimator

    def predict(self, tokens):
        tokens = [(token, "X") for token in tokens]
        x = self.featurizer.process([tokens])[0]
        tags = self.estimator.tag(x)
        return tags
