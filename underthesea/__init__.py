# -*- coding: utf-8 -*-
import os
import sys

__author__ = """Vu Anh"""
__email__ = 'anhv.ict91@gmail.com'

# Check python version
try:
    version_info = sys.version_info
    if version_info < (3, 6, 0):
        raise RuntimeError("underthesea requires Python 3.6 or later")
except Exception:
    pass

###########################################################
# METADATA
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
from underthesea.pipeline.sent_tokenize import sent_tokenize
from underthesea.pipeline.word_tokenize import word_tokenize
from underthesea.pipeline.pos_tag import pos_tag
from underthesea.pipeline.chunking import chunk
from underthesea.pipeline.ner import ner
from underthesea.pipeline.dependency_parse import dependency_parse

try:
    from underthesea.pipeline.classification import classify
except Exception:
    pass
try:
    from underthesea.pipeline.sentiment import sentiment
except Exception:
    pass

__all__ = [
    'sent_tokenize',
    'word_tokenize', 'pos_tag', 'chunk',
    'dependency_parse',
    'ner',
    'classify', 'sentiment',
]
