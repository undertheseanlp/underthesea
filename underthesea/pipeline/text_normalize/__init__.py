from os.path import join, dirname
import joblib


class TextNormalizer:
    def __init__(self, binary_file):
        data = joblib.load(binary_file)
        self.character_map = data["character"]
        self.token_map = data["token"]


text_normalizer = TextNormalizer(join(dirname(__file__), "tn_rules_2022_08_11.bin"))
character_map = text_normalizer.character_map
token_map = text_normalizer.token_map


def character_normalize(text):
    for character_non_standard in character_map:
        character_standard = character_map[character_non_standard]
        text = text.replace(character_non_standard, character_standard)
    return text


def token_normalize(token, use_character_normalize=True):
    """
    normalize each token
    """
    if len(token) > 7:
        return token
    # character normalize
    if use_character_normalize:
        token = character_map(token)
    if token in token_map:
        return token_map[token]
    return token


def text_normalize(text):
    from underthesea.pipeline.word_tokenize.regex_tokenize import tokenize
    tokens = tokenize(text)
    normalized_tokens = [token_normalize(token) for token in tokens]
    normalized_text = " ".join(normalized_tokens)
    return normalized_text
