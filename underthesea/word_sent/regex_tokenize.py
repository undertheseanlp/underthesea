# -*- coding: utf-8 -*-
import re
import sys

from underthesea.feature_engineering.text import Text


def tokenize(text):
    """
    tokenize text for word segmentation

    :param text: raw text input
    :return: tokenize text
    """
    text = Text(text)
    specials = ["==>", "->", "\.\.\.", ">>"]
    digit = "\d+([\.,_]\d+)+"
    email = "(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)"
    web = "^(http[s]?://)?(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+$"
    datetime = [
        "\d{1,2}\/\d{1,2}(\/\d+)?",
        "\d{1,2}-\d{1,2}(-\d+)?",
    ]
    word = "\w+"
    non_word = "[^\w\s]"
    abbreviations = [
        "[A-ZĐ]+\.",
        "Tp\.",
        "Mr\.", "Mrs\.", "Ms\.",
        "Dr\.", "ThS\."
    ]

    patterns = []
    patterns.extend(abbreviations)
    patterns.extend(specials)
    patterns.extend([web, email])
    patterns.extend(datetime)
    patterns.extend([digit, non_word, word])

    patterns = "(" + "|".join(patterns) + ")"
    if sys.version_info < (3, 0):
        patterns = patterns.decode('utf-8')
    tokens = re.findall(patterns, text, re.UNICODE)
    return u" ".join(["%s" % token[0] for token in tokens])
