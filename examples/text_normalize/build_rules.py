from os.path import join, dirname
import joblib
from datetime import datetime
import pandas as pd
from normalize import AnalysableWord

TOKEN_RULE_FILE = join(dirname(__file__), "token_rules.xlsx")
CHARACTER_RULE_FILE = join(dirname(__file__), "character_rules.xlsx")
PROJECT_FOLDER = join(dirname(__file__))


def standardize_token_rule():
    base_norm_dict = {}
    rules_df = pd.read_excel(TOKEN_RULE_FILE, engine='openpyxl')
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
    dict_df.to_excel(TOKEN_RULE_FILE, index=False)


def build_binary_file():
    # load base_norm_dict from rules.xlsx file
    base_norm_dict = {}
    rules_df = pd.read_excel(join(dirname(__file__), TOKEN_RULE_FILE), engine='openpyxl')
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

    character_rules_df = pd.read_excel(CHARACTER_RULE_FILE, engine='openpyxl')
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
    joblib.dump(normalize_map, join(dirname(__file__), "tmp", f"tn_rules_{timestamp}.bin"))


class LatexCharacter:
    def __init__(self, s):
        self.character = s

    def generate_character_cell(self):
        unicode_code = hex(ord(self.character)).upper()
        if len(unicode_code) == 4:
            unicode_code = "00" + unicode_code[2:]
        elif len(unicode_code) == 5:
            unicode_code = "0" + unicode_code[2:]
        cell = ""
        cell += "\\begin{tabular}{@{}c@{}}" + self.character
        cell += "\\\\ \small\\verb|" + unicode_code + "|\end{tabular}"
        return cell

    def __str__(self):
        return self.character


def build_character_rule_latex_table():
    character_rules_df = pd.read_excel(CHARACTER_RULE_FILE, engine='openpyxl')
    CHARACTER_RULE_LATEX_TABLE = join(PROJECT_FOLDER, "reports", "latex", "character_rule_latex_table.txt")
    content = """
\\begin{table}
\centering
\\begin{tabular}{c c c}
\hline
\\textbf{ID} & \\textbf{None Standard} & \\textbf{Standard} \\\\
\hline
""".strip()
    content += '\n'
    for index, row in character_rules_df.iterrows():
        non_standard = LatexCharacter(row[0])
        standard = LatexCharacter(row[1])
        line = f"{index + 1} & \n"
        line += f"{non_standard.generate_character_cell()} & \n{standard.generate_character_cell()} \\\\"
        content += line + '\n'
        content += '\hline' + '\n'
    content += """\end{tabular}
\caption{Character standardization rules}
\label{table:character_rules}
\end{table}
""".strip()
    with open(CHARACTER_RULE_LATEX_TABLE, "w") as f:
        f.write(content)


if __name__ == '__main__':
    # standardize_token_rule()
    build_binary_file()
    # build_character_rule_latex_table()
