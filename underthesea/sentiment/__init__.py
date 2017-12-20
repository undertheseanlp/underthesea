from underthesea.sentiment import bank


def classify(X, domain=None):
    if X == "":
        return None
    if domain == 'bank':
        return bank.sentiment(X)
