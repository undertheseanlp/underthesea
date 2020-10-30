# -*- coding: utf-8 -*-
import re
import sys
from underthesea.feature_engineering.text import Text

#################################################
# PRIORITY 1                                    #
#################################################
abbreviations = [
    # & at middle of word (e.g. H&M)
    r"[A-ZĐ]+&[A-ZĐ]+",
    # dot at middle of word
    r"T\.Ư",
    # dot at the end of word
    r"[A-ZĐ]+\.(?!$)",
    r"Tp\.",
    r"Mr\.", "Mrs\.", "Ms\.",
    r"Dr\.", "ThS\."
]
abbreviations = "(?P<abbr>(" + "|".join(abbreviations) + "))"

specials = [
    r"=\>",
    r"==>",
    r"->",
    r"\.\.\.",
    r">>",
]
specials = "(?P<special>(" + "|".join(specials) + "))"

#################################################
# PRIORITY 2                                    #
#################################################

# urls pattern from nltk
# https://www.nltk.org/_modules/nltk/tokenize/casual.html
# with Vu Anh's modified to match fpt protocol
url = r"""             # Capture 1: entire matched URL
  (?:
  (ftp|http)s?:               # URL protocol and colon
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
url = "(?P<url>" + url + ")"

email = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
email = "(?P<email>" + email + ")"

datetime = [
    r"\d{1,2}\/\d{1,2}(\/\d+)?",
    r"\d{1,2}-\d{1,2}(-\d+)?",
    r"\d{1,2}\.\d{1,2}\.\d+",
]
datetime = "(?P<datetime>(" + "|".join(datetime) + "))"

name = [

    r"\d+[A-Z]+"    # case
                    # 4K
    # r"\w+\-\w+"   # [WIP] deprecated
                    # case
                    #   F-16, Su-34, Rolls-Royce
                    # conflict with
                    #   2010-2015
                    #   Moscow-Washington
                    # issue #290
]
name = "(?P<name>(" + "|".join(name) + "))"

number = r"\d+([\.,_]\d+)?"
number = "(?P<number>" + number + ")"

emoji = [
    r":\)\)*",
    r"=\)\)+",
    r"♥‿♥",
    r":D+(?=\s)",  # :D
    r":D+(?=$)",   # special case: Đạo diễn :Dương Tuấn Anh
    r"<3"          # heart
]
emoji = "(?P<emoji>(" + "|".join(emoji) + "))"

symbols = [
    r"\+",
    r"-",
    r"×",
    r"÷",
    r":+",
    r"%",
    r"%",
    r"\$",
    r"\>",
    r"\<",
    r"=",
    r"\^",
    r"_",
    r":+"
]
symbols = "(?P<sym>(" + "|".join(symbols) + "))"

#################################################
# PRIORITY 3                                    #
#################################################
word = r"(?P<word>\w+)"

punct = [
    r"\.",
    r"\,",
    r"\(",
    r"\)"
]
punct = "(?P<punct>(" + "|".join(punct) + "))"

non_word = r"(?P<non_word>[^\w\s])"

# Caution: order is very important for regex
patterns = [
    abbreviations,      # Priority 1
    specials,
    url,                # Priority 2
    email,              # datetime must be before number
    datetime,
    name,
    number,
    emoji,
    symbols,
    word,               # Priority 3
    punct,              # word and non_word must be last
    non_word
]

patterns = r"(" + "|".join(patterns) + ")"
if sys.version_info < (3, 0):
    patterns = patterns.decode('utf-8')
patterns = re.compile(patterns, re.VERBOSE | re.UNICODE)


def extract_match(m):
    for k, v in m.groupdict().items():
        if v is not None:
            return v, k


def tokenize(text, format=None):
    """
    tokenize text for word segmentation

    :param text: raw text input
    :return: tokenize text
    """
    text = Text(text)
    text = text.replace("\t", " ")
    matches = [m for m in re.finditer(patterns, text)]
    tokens = [extract_match(m) for m in matches]
    tokens = [token[0] for token in tokens]
    if format == "text":
        return " ".join(tokens)
    else:
        return tokens
