import joblib

from underthesea.transformer.tagged import TaggedTransformer


class CRFSequenceTagger:
    def __init__(self, features=None):
        self.features = features
        self.transformer = TaggedTransformer(features)

    def forward(self, samples, contains_labels=False):
        return self.transformer.transform(samples, contains_labels)

    def save(self, path):
        print("save features")
        joblib.dump(self.features, path)
        pass
