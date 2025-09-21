from .module import VietnameseTextNormalizer


def normalize(word):
    text = VietnameseTextNormalizer.Normalize(word)
    return text
