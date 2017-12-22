from underthesea.sentiment import bank


def sentiment(X, domain=None):
    if X == "":
        return None
    if domain == 'bank':
        return bank.sentiment(X)
