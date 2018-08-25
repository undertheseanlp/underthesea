import joblib
from os.path import join, dirname
import sys

sys.path.insert(0, dirname(__file__))

bank_classification = {}


def classify(X):
    global bank_classification
    if "x_transform" not in bank_classification:
        bank_classification["x_transform"] = joblib.load(join(dirname(__file__), "tfidf.transformer.bin"))
    if "y_transform" not in bank_classification:
        bank_classification["y_transform"] = joblib.load(join(dirname(__file__), "label.transformer.bin"))
    if "estimator" not in bank_classification:
        bank_classification["estimator"] = joblib.load(join(dirname(__file__), "model.bin"))
    x_transform = bank_classification["x_transform"]
    y_transform = bank_classification["y_transform"]
    estimator = bank_classification["estimator"]
    if isinstance(X, list):
        return y_transform.inverse_transform(
            estimator.predict(x_transform.transform(X)))
    else:
        return y_transform.inverse_transform(
            estimator.predict(x_transform.transform([X])))[0]
