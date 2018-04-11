# -*- coding: utf-8 -*-
import os
import sys

__author__ = """Vu Anh"""
__email__ = 'brother.rain.1024@gmail.com'

###########################################################
# Metadata
###########################################################

# Version
try:
    version_file = os.path.join(os.path.dirname(__file__), 'VERSION')
    with open(version_file, 'r') as infile:
        __version__ = infile.read().strip()
except NameError:
    __version__ = 'unknown (running code interactively?)'
except IOError as ex:
    __version__ = "unknown (%s)" % ex



###########################################################
# TOP-LEVEL MODULES
###########################################################
if sys.version_info >= (3, 0):
    from underthesea.word_tokenize import word_tokenize
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
    from word_tokenize import word_tokenize
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
