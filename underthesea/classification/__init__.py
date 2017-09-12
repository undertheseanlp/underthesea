from underthesea.classification.model_fasttext import FastTextPredictor


def classify(X):
    if X == "":
        return None
    else:
        clf = FastTextPredictor.Instance()
        return clf.predict(X)
