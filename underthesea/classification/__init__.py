from underthesea.classification.model_fasttext import FastTextPredictor
from underthesea.classification import bank


def classify(X, domain=None):
    if X == "":
        return None
    if domain == 'bank':
        return bank.classify(X)
    # domain is general
    clf = FastTextPredictor.Instance()
    return clf.predict(X)
