import sys

import unicodedata


def Text(text):
    """ provide a wrapper for python string
    map byte to str (python 3)
    map str to unicode (python 2)
    all string in utf-8 encoding
    normalize string to NFC
    """
    if not is_unicode(text):
       text = text.decode("utf-8")
    text = unicodedata.normalize("NFC", text)
    return text


def is_unicode(text):
    if sys.version_info >= (3, 0):
        unicode_type = str
    else:
        unicode_type = unicode
    return type(text) == unicode_type
