from .character_normalize import normalize_characters_in_text
from .text_normalizer import token_map


def token_normalize(token, use_character_normalize=True):
    """
    normalize each token
    """
    if len(token) > 6:
        return token
    # character normalize
    if use_character_normalize:
        token = normalize_characters_in_text(token)
    if token in token_map:
        return token_map[token]
    return token
