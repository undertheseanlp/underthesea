from underthesea import classify


def sentiment(X, domain=""):
    aspects = classify(X, domain=domain)
    polarity = polarity(X)
    s = [] # merge aspects with polarity
    return s
