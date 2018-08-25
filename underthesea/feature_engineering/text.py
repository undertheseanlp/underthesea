import unicodedata


def Text(text):
    """ provide a wrapper for python string
    map byte to str (python 3)
    all string in utf-8 encoding
    normalize string to NFC
    """
    if not is_unicode(text):
        text = text.decode("utf-8")
    text = unicodedata.normalize("NFC", text)
    return text


def is_unicode(text):
    return type(text) == str
