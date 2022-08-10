import pandas as pd

from underthesea.feature_engineering.normalize import AnalysableWord

base_norm_dict = {}
rules_df = pd.read_excel("rules.xlsx", engine='openpyxl')
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
dict_df.to_excel("rules.xlsx", index=False)
