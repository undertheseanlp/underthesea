from os.path import join
from pathlib import Path
import joblib
import pycrfsuite

from underthesea.transformer.tagged import TaggedTransformer


class CRFSequenceTagger:
    def __init__(self, features=None):
        self.features = features
        self.estimator = None
        self.transformer = None

    def forward(self, samples, contains_labels=False):
        if not self.transformer:
            self.transformer = TaggedTransformer(self.features)
        return self.transformer.transform(samples, contains_labels)

    def save(self, path):
        print("save features")
        joblib.dump(self.features, path)

    def load(self, base_path):
        print(base_path)
        model_path = str(Path(base_path) / "model.bin")
        estimator = pycrfsuite.Tagger()
        estimator.open(model_path)
        features = joblib.load(join(base_path, "features.bin"))
        transformer = TaggedTransformer(features)
        self.transformer = transformer
        self.estimator = estimator

    def predict(self, tokens):
        tokens = [(token, "X") for token in tokens]
        x = self.transformer.transform([tokens])[0]
        tags = self.estimator.tag(x)
        return tags
