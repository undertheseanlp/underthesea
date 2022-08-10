import viet_text_tools


def normalize(word):
    text = viet_text_tools.normalize_diacritics(word)
    return text
