import re


def tokenize(text):
    """
    tokenize text for word segmentation

    :param text: raw text input
    :return: tokenize text
    """
    specials = ["==>", "->", "\.\.\.", ">>"]
    digit = "\d+([\.,_]\d+)+"
    email = "(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)"
    web = "^(http[s]?://)?(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+$"
    datetime = "\d{,2}\/\d{,2}\/\d{,4}"
    word = "\w+"
    non_word = "[^\w\s]"

    patterns = []
    patterns.extend(specials)
    patterns.extend([web, email])
    patterns.extend([datetime])
    patterns.extend([digit, non_word, word])

    patterns = "(" + "|".join(patterns) + ")"
    tokens = re.findall(patterns, text, re.UNICODE)
    return u" ".join(["%s" % token[0] for token in tokens])
