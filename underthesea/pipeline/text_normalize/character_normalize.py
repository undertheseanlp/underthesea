import unicodedata

from .text_normalizer import character_map


def character_normalize(text):
    for character_non_standard in character_map:
        character_standard = character_map[character_non_standard]
        text = text.replace(character_non_standard, character_standard)
    return text


def utf8_normalize(text):
    if not is_unicode(text):
        text = text.decode("utf-8")
    text = unicodedata.normalize("NFC", text)
    return text


def normalize_characters_in_text(text):
    text = utf8_normalize(text)
    text = character_normalize(text)
    return text


def is_unicode(text):
    return type(text) == str
