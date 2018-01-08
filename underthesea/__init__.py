# -*- coding: utf-8 -*-

__author__ = """Vu Anh"""
__email__ = 'brother.rain.1024@gmail.com'
__version__ = '1.1.5'

import sys

if sys.version_info >= (3, 0):
    from underthesea.word_sent import word_sent
    from underthesea.pos_tag import pos_tag
    from underthesea.chunking import chunk
    from underthesea.ner import ner
    try:
        from underthesea.classification import classify
    except Exception as e:
        pass
    try:
        from underthesea.sentiment import sentiment
    except Exception as e:
        pass
else:
    from word_sent import word_sent
    from pos_tag import pos_tag
    from chunking import chunk
    from ner import ner
    try:
        from classification import classify
    except Exception as e:
        pass

    try:
        from sentiment import sentiment
    except Exception as e:
        pass


def info(version):
    """Show information about underthesea package

    :param str version: version of package
    :type version: str
    """
    print(version)
    print("underthesea is a Vietnamese NLP Toolkit")
