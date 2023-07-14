from os.path import join, dirname

import joblib


class TextNormalizer:
    def __init__(self, binary_file):
        data = joblib.load(binary_file)
        self.character_map = data["character"]
        self.token_map = data["token"]


# rule_file = "tn_rules_2022_08_11.bin"
rule_file = "tn_rules_2023_07_14.bin"
text_normalizer = TextNormalizer(join(dirname(__file__), rule_file))
character_map = text_normalizer.character_map
token_map = text_normalizer.token_map
text_normalizer = TextNormalizer(join(dirname(__file__), rule_file))
character_map = text_normalizer.character_map
token_map = text_normalizer.token_map
