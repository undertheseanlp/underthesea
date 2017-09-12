# -*- coding: utf-8 -*-
import unicodedata
from analyze_characters import analyze_characters, get_utf8_number, get_unicode_number

inverse_mapping_table = {
    # 0110
    u"Đ": [
        u"Ð",  # 00D0
        u"Ɖ",  # 0189
        u"ᴆ",  # 1D06
    ]
}
mapping_table = {}
for key, characters in inverse_mapping_table.iteritems():
    for character in characters:
        mapping_table[character] = key


def map_character_to_tcvn(c):
    if c in mapping_table:
        return mapping_table[c]
    else:
        return c


def map_text_to_tcvn(text):
    """
    @param unicode text: converted to normalize nfc form
    """
    return [map_character_to_tcvn(c) for c in text]


def vietnamese_normalize(text):
    """
    @param text: unicode
    """
    text = unicodedata.normalize("NFC", text)
    text = map_text_to_tcvn(text)
    return text


if __name__ == '__main__':
    text = u"ĐƉᴆ"
    analyze_characters(text)

    print "\nAfter normalize\n"

    analyze_characters(vietnamese_normalize(text))
