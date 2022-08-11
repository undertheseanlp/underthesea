from .token_normalize import token_normalize
from underthesea.pipeline.word_tokenize.regex_tokenize import tokenize


def text_normalize(text, tokenizer='underthesea'):
    """

    Args:
        tokenizer (str): space or underthesea
    """
    if tokenizer == 'underthesea':
        tokens = tokenize(text)
    else:
        tokens = text.split(" ")
    normalized_tokens = [token_normalize(token) for token in tokens]
    normalized_text = " ".join(normalized_tokens)
    return normalized_text
