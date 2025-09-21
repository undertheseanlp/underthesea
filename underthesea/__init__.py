"""
Underthesea
"""
# -*- coding: utf-8 -*-
import os
import sys
from functools import lru_cache


__author__ = """Vu Anh"""
__email__ = 'anhv.ict91@gmail.com'

# Check python version
try:
    version_info = sys.version_info
    if version_info < (3, 9, 0):
        raise RuntimeError("underthesea requires Python 3.9 or later")
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
from .pipeline.sent_tokenize import sent_tokenize
from .pipeline.text_normalize import text_normalize
from .pipeline.word_tokenize import word_tokenize
from .pipeline.pos_tag import pos_tag
from .pipeline.chunking import chunk
from .pipeline.ner import ner

optional_imports = {
    'classify': 'underthesea.pipeline.classification',
    'sentiment': 'underthesea.pipeline.sentiment',
    'lang_detect': 'underthesea.pipeline.lang_detect',
    'dependency_parse': 'underthesea.pipeline.dependency_parse'
}


@lru_cache(maxsize=None)
def get_optional_import(module_name, object_name):
    try:
        module = __import__(module_name, fromlist=[object_name])
        return getattr(module, object_name)
    except ImportError:
        return None


for name, module in optional_imports.items():
    globals()[name] = get_optional_import(module, name)

__all__ = [
    'sent_tokenize',
    'text_normalize',
    'word_tokenize',
    'pos_tag',
    'chunk',
    'ner',
    'lang_detect',
    'classify',
    # 'sentiment',
    'dependency_parse'
]
