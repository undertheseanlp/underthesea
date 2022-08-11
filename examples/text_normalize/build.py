from os.path import join, dirname
import joblib
from datetime import datetime
import pandas as pd
from examples.text_normalize.normalize import AnalysableWord

RULE_FILE = join(dirname(__file__), "token_rules.xlsx")


def standardize_rule():
    base_norm_dict = {}
    rules_df = pd.read_excel(RULE_FILE, engine='openpyxl')
    for key, item in rules_df.iterrows():
        word = item[0]
        normalize = item[1]
        base_norm_dict[word] = normalize

    base_norm_data = []
    for key in base_norm_dict:
        word = AnalysableWord(base_norm_dict[key])
        base_norm_data.append({
            "word": key,
            "normalize": base_norm_dict[key],
            "group": word.group
        })
    dict_df = pd.DataFrame(base_norm_data)
    dict_df = dict_df.sort_values(by=["group", "normalize", "word"])
    dict_df.to_excel(RULE_FILE, index=False)


def build_binary_file():
    # load base_norm_dict from rules.xlsx file
    base_norm_dict = {}
    rules_df = pd.read_excel(join(dirname(__file__), RULE_FILE), engine='openpyxl')
    for key, item in rules_df.iterrows():
        word = item[0]
        normalize = item[1]
        base_norm_dict[word] = normalize

    norm_dict = base_norm_dict.copy()
    for key in base_norm_dict:
        # add capitalize rules
        new_key = key.capitalize()
        new_value = base_norm_dict[key].capitalize()
        norm_dict[new_key] = new_value
        # add uppercase rules
        new_key = key.upper()
        new_value = base_norm_dict[key].upper()
        norm_dict[new_key] = new_value

    character_rules_df = pd.read_excel(join(dirname(__file__), "character_rules.xlsx"), engine='openpyxl')
    character_map = {}
    for id, row in character_rules_df.iterrows():
        non_standard = row[0]
        standard = row[1]
        character_map[non_standard] = standard

    normalize_map = {
        "character": character_map,
        "token": norm_dict
    }
    timestamp = datetime.now().strftime("%Y_%m_%d")
    joblib.dump(normalize_map, join(dirname(__file__), "data", f"tn_rules_{timestamp}.bin"))


if __name__ == '__main__':
    standardize_rule()
    build_binary_file()
