from os.path import join, dirname
import pandas as pd

# open token_rules file using pandas
rules_df = pd.read_excel(join(dirname(__file__), "token_rules.xlsx"), engine='openpyxl')


def rule_01_normalize_column_must_be_striped():
    # check column normalize must be striped
    global rules_df
    words = list(rules_df["normalize"])
    for word in words:
        if word.strip() != word:
            print("[ERR] Column normalize must be striped", word)


if __name__ == "__main__":
    rule_01_normalize_column_must_be_striped()
