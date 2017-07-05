# -*- coding: utf-8 -*-

__author__ = """Vu Anh"""
__email__ = 'brother.rain.1024@gmail.com'
__version__ = '1.1.1'

import sys

if sys.version_info >= (3, 0):
    from underthesea.word_sent.word_sent import word_sent
    from underthesea.pos_tag.pos_tag import pos_tag
    from underthesea.chunking.chunk import chunk
else:
    from word_sent.word_sent import word_sent
    from pos_tag.pos_tag import pos_tag
    from chunking.chunk import chunk


def info(version):
    """Show information about underthesea package

    :param str version: version of package
    :type version: str
    """
    print(version)
    print("underthesea is a Vietnamese NLP Toolkit")
