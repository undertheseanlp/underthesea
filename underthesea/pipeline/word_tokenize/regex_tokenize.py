# -*- coding: utf-8 -*-
import re
import sys

from underthesea.pipeline.text_normalize import token_normalize
from underthesea.pipeline.text_normalize.character_normalize import normalize_characters_in_text

UPPER = "[" + "".join([
    "A-Z",
    "ÀÁẢÃẠ",
    "ĂẰẮẲẴẶ",
    "ÂẦẤẨẪẬ",
    "Đ",
    "ÈÉẺẼẸ",
    "ÊỀẾỂỄỆ",
    "ÌÍỈĨỊ",
    "ÒÓỎÕỌ",
    "ÔỒỐỔỖỘ",
    "ƠỜỚỞỠỢ",
    "ÙÚỦŨỤ",
    "ƯỪỨỬỮỰ",
    "ỲÝỶỸỴ"
]) + "]"
LOWER = UPPER.lower()
W = "[" + UPPER[1:-1] + LOWER[1:-1] + "]"  # upper and lower

VIETNAMESE_CHARACTERS_UPPER = "ABCDEFGHIJKLMNOPQRSTUVXYZ" \
                              + "ÀÁẢÃẠ" + "ĂẰẮẲẴẶ" + "ÂẦẤẨẪẬ" \
                              + "Đ" \
                              + "ÈÉẺẼẸ" + "ÊỀẾỂỄỆ" \
                              + "ÌÍỈĨỊ" \
                              + "ÒÓỎÕỌ" + "ÔỒỐỔỖỘ" + "ƠỜỚỞỠỢ" \
                              + "ÙÚỦŨỤ" + "ƯỪỨỬỮỰ" \
                              + "ỲÝỶỸỴ"
VIETNAMESE_CHARACTERS_LOWER = VIETNAMESE_CHARACTERS_UPPER.lower()
VIETNAMESE_VOWELS_UPPER = "AEIOU" \
                          + "ÀÁẢÃẠ" + "ĂẰẮẲẴẶ" + "ÂẦẤẨẪẬ" \
                          + "ÈÉẺẼẸ" + "ÊỀẾỂỄỆ" \
                          + "ÌÍỈĨỊ" \
                          + "ÒÓỎÕỌ" + "ÔỒỐỔỖỘ" + "ƠỜỚỞỠỢ" \
                          + "ÙÚỦŨỤ" + "ƯỪỨỬỮỰ" \
                          + "ỲÝỶỸỴ"
VIETNAMESE_VOWELS_LOWER = VIETNAMESE_VOWELS_UPPER.lower()

#################################################
# PRIORITY 1                                    #
#################################################
specials = [
    r"=\>",
    r"==>",
    r"->",
    r"\.{2,}",
    r"-{2,}",
    r">>",
    r"\d+x\d+",  # dimension: 3x4
    r"v\.v\.\.\.",
    r"v\.v\.",
    r"v\.v",
    r"°[CF]"
]
specials = "(?P<special>(" + "|".join(specials) + "))"

abbreviations = [
    r"[A-ZĐ]+&[A-ZĐ]+",  # & at middle of word (e.g. H&M)
    r"T\.Ư",  # dot at middle of word
    f"{UPPER}+(?:\.{W}+)+\.?",
    f"{W}+['’]{W}+",  # ' ’ at middle of word
    # e.g. H'Mông, xã N’Thôn Hạ
    r"[A-ZĐ]+\.(?!$)",  # dot at the end of word
    r"Tp\.",
    r"Mr\.", "Mrs\.", "Ms\.",
    r"Dr\.", "ThS\.", "Th.S", "Th.s",
    r"e-mail",  # - at middle of word
    r"\d+[A-Z]+\d*-\d+",  # vehicle plates
    # e.g. 43H-0530
    r"NĐ-CP"
]
abbreviations = "(?P<abbr>(" + "|".join(abbreviations) + "))"

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

phone = [
    r"\d{2,}-\d{3,}-\d{3,}"  # e.g. 03-5730-2357
    # very careful, it's easy to conflict with datetime
]
phone = "(?P<phone>(" + "|".join(phone) + "))"

datetime = [
    # date
    r"\d{1,2}\/\d{1,2}\/\d+",  # e.g. 02/05/2014
    r"\d{1,2}\/\d{1,4}",  # e.g. 02/2014
    #   [WIP] conflict with number 1/2 (a half)
    r"\d{1,2}-\d{1,2}-\d+",  # e.g. 02-03-2014
    r"\d{1,2}-\d{1,4}",  # e.g. 08-2014
    #   [WIP] conflict with range 5-10 (from 5 to 10)
    r"\d{1,2}\.\d{1,2}\.\d+",  # e.g. 20.08.2014
    r"\d{4}\/\d{1,2}\/\d{1,2}",  # e.g. 2014/08/20
    r"\d{2}:\d{2}:\d{2}"  # time
    # e.g. 10:20:50 (10 hours, 20 minutes, 50 seconds)
]
datetime = "(?P<datetime>(" + "|".join(datetime) + "))"

name = [
    r"\d+[A-Z]+\d+",
    r"\d+[A-Z]+"  # e.g. 4K
]
name = "(?P<name>(" + "|".join(name) + "))"

number = [
    r"\d+(?:\.\d+)+,\d+",  # e.g. 4.123,2
    r"\d+(?:\.\d+)+",  # e.g. 60.542.000
    r"\d+(?:,\d+)+",  # e.g. 100,000,000
    r"\d+(?:[\.,_]\d+)?",  # 123
]
number = "(?P<number>(" + "|".join(number) + "))"

emoji = [
    r":\)\)*",
    r"=\)\)+",
    r"♥‿♥",
    r":D+(?=\s)",  # :D
    r":D+(?=$)",  # special e.g. Đạo diễn :Dương Tuấn Anh
    r"<3"  # heart
]
emoji = "(?P<emoji>(" + "|".join(emoji) + "))"

punct = [
    r"\.",
    r"\,",
    r"\(",
    r"\)",
    r"ʺ"  # Modifier Letter Double Prime symbol (U+02BA)
]
punct = "(?P<punct>(" + "|".join(punct) + "))"

#################################################
# PRIORITY 3                                    #
#################################################
word = r"(?P<word>\w+)"

word_hyphen = [
    r"(?<=\b)\w+\-[\w+-]*\w+"  # before word_hyphen must be word boundary
    # case to notice: 1.600m-2.000m
]
word_hyphen = "(?P<word_hyphen>(" + "|".join(word_hyphen) + "))"

symbol = [
    r"\+",
    r"×",
    r"-",
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
symbol = "(?P<sym>(" + "|".join(symbol) + "))"

non_word = r"(?P<non_word>[^\w\s])"

# Caution: order is very important for regex
patterns = [
    specials,  # Priority 1
    abbreviations,
    url,  # Priority 2
    email,
    phone,
    datetime,  # datetime must be before number
    name,
    number,
    emoji,
    punct,
    word_hyphen,  # Priority 3
    word,
    symbol,
    non_word  # non_word must be last
]

patterns = r"(" + "|".join(patterns) + ")"
if sys.version_info < (3, 0):
    patterns = patterns.decode('utf-8')
patterns = re.compile(patterns, re.VERBOSE | re.UNICODE)


def extract_match(m):
    for k, v in m.groupdict().items():
        if v is not None:
            return v, k


def tokenize(text, format=None, tag=False, use_character_normalize=True, use_token_normalize=True):
    """
    tokenize text for word segmentation

    Args:
        use_token_normalize:
        use_character_normalize:
        tag:
        format:
    """
    if use_character_normalize:
        text = normalize_characters_in_text(text)
    matches = [m for m in re.finditer(patterns, text)]
    tokens = [extract_match(m) for m in matches]

    if tag:
        return tokens

    tokens = [token[0] for token in tokens]
    if use_token_normalize:
        tokens = [token_normalize(_, use_character_normalize=use_character_normalize) for _ in tokens]

    if format == "text":
        return " ".join(tokens)

    return tokens
