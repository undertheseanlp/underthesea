from .token_normalize import token_normalize
from underthesea.pipeline.word_tokenize.regex_tokenize import tokenize


def text_normalize(text):
    tokens = tokenize(text)
    normalized_tokens = [token_normalize(token) for token in tokens]
    normalized_text = " ".join(normalized_tokens)
    return normalized_text
