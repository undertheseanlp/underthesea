import joblib
from os.path import join, dirname
import sys

sys.path.insert(0, dirname(__file__))

bank_sentiment = {}


def sentiment(X):
    global bank_sentiment
    if "x_transform" not in bank_sentiment:
        bank_sentiment["x_transform"] = joblib.load(join(dirname(__file__), "count.transformer.bin"))
    if "y_transform" not in bank_sentiment:
        bank_sentiment["y_transform"] = joblib.load(join(dirname(__file__), "label.transformer.bin"))
    if "estimator" not in bank_sentiment:
        bank_sentiment["estimator"] = joblib.load(join(dirname(__file__), "model.bin"))
    x_transform = bank_sentiment["x_transform"]
    y_transform = bank_sentiment["y_transform"]
    estimator = bank_sentiment["estimator"]
    if isinstance(X, list):
        return y_transform.inverse_transform(
            estimator.predict(x_transform.transform(X)))
    else:
        return y_transform.inverse_transform(
            estimator.predict(x_transform.transform([X])))[0]
