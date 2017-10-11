# -*- coding: utf-8 -*-

__author__ = """Vu Anh"""
__email__ = 'brother.rain.1024@gmail.com'
__version__ = '1.1.4'

import sys

if sys.version_info >= (3, 0):
    from underthesea.word_sent import word_sent
    from underthesea.pos_tag import pos_tag
    from underthesea.chunking.chunk import chunk
    from underthesea.classification import classify
    from underthesea.ner import ner
else:
    from word_sent import word_sent
    from pos_tag import pos_tag
    from chunking.chunk import chunk
    from classification import classify
    from ner import ner


def info(version):
    """Show information about underthesea package

    :param str version: version of package
    :type version: str
    """
    print(version)
    print("underthesea is a Vietnamese NLP Toolkit")
