"""
Underthesea
"""
# -*- coding: utf-8 -*-
import sys
from functools import cache

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

from underthesea.version import __version__

###########################################################
# TOP-LEVEL MODULES
###########################################################
from .pipeline.chunking import chunk
from .pipeline.ner import ner
from .pipeline.pos_tag import pos_tag
from .pipeline.sent_tokenize import sent_tokenize
from .pipeline.text_normalize import text_normalize
from .pipeline.word_tokenize import word_tokenize

optional_imports = {
    'classify': 'underthesea.pipeline.classification',
    'sentiment': 'underthesea.pipeline.sentiment',
    'lang_detect': 'underthesea.pipeline.lang_detect',
    'dependency_parse': 'underthesea.pipeline.dependency_parse',
    'translate': 'underthesea.pipeline.translate',
    'agent': 'underthesea.agent',
}


@cache
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
    'dependency_parse',
    'translate',
    'agent',
]
