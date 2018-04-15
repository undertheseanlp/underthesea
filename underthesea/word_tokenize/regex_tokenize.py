# -*- coding: utf-8 -*-
import re
import sys

from underthesea.feature_engineering.text import Text

specials = ["==>", "->", "\.\.\.", ">>", "=\)\)"]
digit = "\d+([\.,_]\d+)+"
email = "[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"

# urls pattern from nltk
# https://www.nltk.org/_modules/nltk/tokenize/casual.html
urls = r"""             # Capture 1: entire matched URL
  (?:
  https?                # URL protocol and colon
    (?:
      /{1,3}            # 1-3 slashes
      |                 #   or
      [a-z0-9%]         # Single letter or digit or '%'
                        # (Trying not to match e.g. "URI::Escape")
    )
    |                   #   or
                        # looks like domain name followed by a slash:
    [a-z0-9.\-]+[.]
    (?:[a-z]{2,13})
    /
  )
  (?:                                  # One or more:
    [^\s()<>{}\[\]]+                   # Run of non-space, non-()<>{}[]
    |                                  #   or
    \([^\s()]*?\([^\s()]+\)[^\s()]*?\) # balanced parens, one level deep: (...(...)...)
    |
    \([^\s]+?\)                        # balanced parens, non-recursive: (...)
  )+
  (?:                                  # End with:
    \([^\s()]*?\([^\s()]+\)[^\s()]*?\) # balanced parens, one level deep: (...(...)...)
    |
    \([^\s]+?\)                        # balanced parens, non-recursive: (...)
    |                                  #   or
    [^\s`!()\[\]{};:'".,<>?«»“”‘’]     # not a space or one of these punct chars
  )
  |                        # OR, the following to match naked domains:
  (?:
    (?<!@)                 # not preceded by a @, avoid matching foo@_gmail.com_
    [a-z0-9]+
    (?:[.\-][a-z0-9]+)*
    [.]
    (?:[a-z]{2,13})
    \b
    /?
    (?!@)                  # not succeeded by a @,
                           # avoid matching "foo.na" in "foo.na@example.com"
  )
"""
datetime = [
    "\d{1,2}\/\d{1,2}(\/\d+)?",
    "\d{1,2}-\d{1,2}(-\d+)?",
]
word = "\w+"
non_word = "[^\w\s]"
abbreviations = [
    "[A-ZĐ]+\.",
    "Tp\.",
    "Mr\.", "Mrs\.", "Ms\.",
    "Dr\.", "ThS\."
]
patterns = []
patterns.extend(abbreviations)
patterns.extend(specials)
patterns.extend([urls])
patterns.extend([email])
patterns.extend(datetime)
patterns.extend([digit, non_word, word])
patterns.extend([digit, word])

patterns = "(" + "|".join(patterns) + ")"
if sys.version_info < (3, 0):
    patterns = patterns.decode('utf-8')


def tokenize(text):
    """
    tokenize text for word segmentation

    :param text: raw text input
    :return: tokenize text
    """
    text = Text(text)
    text = text.replace("\t", " ")
    tokens = re.findall(patterns, text, re.UNICODE)
    return u" ".join(["%s" % token[0] for token in tokens])
