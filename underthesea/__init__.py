# -*- coding: utf-8 -*-
import os

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
from underthesea.sent_tokenize import sent_tokenize
from underthesea.word_tokenize import word_tokenize
from underthesea.pos_tag import pos_tag
from underthesea.chunking import chunk
from underthesea.ner import ner

try:
    from underthesea.classification import classify
except Exception:
    pass
try:
    from underthesea.sentiment import sentiment
except Exception:
    pass

__all__ = [
    'sent_tokenize',
    'word_tokenize', 'pos_tag', 'chunk', 'ner',
    'classify', 'sentiment'
]
