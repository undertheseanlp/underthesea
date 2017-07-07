import sys

import unicodedata


def Text(text):
    """ provide a wrapper for python string
    """
    if sys.version_info >= (3, 0):
        unicode = str
    else:
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
