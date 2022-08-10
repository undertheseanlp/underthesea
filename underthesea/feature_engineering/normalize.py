from datetime import datetime
from os.path import join
from underthesea.file_utils import UNDERTHESEA_FOLDER
from underthesea.pipeline.word_tokenize import tokenize
import pandas as pd

CORPUS_FOLDER = join(UNDERTHESEA_FOLDER, "datasets", "LTA")
TOKENS_ANALYSE_FILE = join(CORPUS_FOLDER, "tokens_analyze.txt")
SYMBOLS_ANALYSE_FILE = join(CORPUS_FOLDER, "symbols_analyze.txt")


def analyze_token_and_symbol():
    count_token_dict = {}
    count_symbol_dict = {}
    filename = join(CORPUS_FOLDER, "VNTQcorpus-big.txt")
    for i, line in enumerate(open(filename)):
        tokens = tokenize(line.strip())
        for token in tokens:
            if token not in count_token_dict:
                count_token_dict[token] = 0
            count_token_dict[token] += 1
        for symbol in line:
            if symbol not in count_symbol_dict:
                count_symbol_dict[symbol] = 0
            count_symbol_dict[symbol] += 1
        # if i > 10000:
        #     break
        if i % 100000 == 0:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(f"{current_time} {i}")
        # if i > 300000:
        #     break
    return count_token_dict, count_symbol_dict


def write_analyzes():
    count_token_dict, count_symbol_dict = analyze_token_and_symbol()
    print('finish tokenize')
    tokens = sorted(count_token_dict.items(), key=lambda x: -x[1])
    with open(TOKENS_ANALYSE_FILE, 'w') as f:
        for item in tokens:
            s = f"{item[0]}\t\t{item[1]}"
            print(s)
            f.write(s + "\n")
    symbols = sorted(count_symbol_dict.items(), key=lambda x: -x[1])
    with open(SYMBOLS_ANALYSE_FILE, 'w') as f:
        for item in symbols:
            s = f"{item[0]}\t\t{item[1]}"
            print(s)
            f.write(s + "\n")


# load base_norm_dict from rules.xlsx file
base_norm_dict = {}
rules_df = pd.read_excel("rules.xlsx", engine='openpyxl')
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


def normalize(s):
    if s in norm_dict:
        return norm_dict[s]
    return s


def count_normalize():
    total_count = 0
    normalized_count = 0
    with open(TOKENS_ANALYSE_FILE) as f:
        for line in f:
            total_count += 1
            word, freq = line.split("\t\t")
            freq = freq.strip()
            print(word, ":", freq)
            if normalize(word) == word:
                normalized_count += 1
            else:
                print("NORMALIZE")
    print("Total words: ", total_count)
    print("Normalized words: ", normalized_count)


syllable_map_r = {
    "a": "ảáảạạạàăãáẵằắăầâấậââẩâẫ",
    "e": "èêẽếềếẻẹééé",
    "o": "ơợốốốớớồôõòóỏọởỡỗổỡởởộ",
    "i": "ỉíìịĩĩ",
    "u": "ûửúủụựưừũùùứùưữ",
    "y": "ỷỳỵỵỹý"
}
syllable_map = {}
for key in syllable_map_r:
    items = syllable_map_r[key]
    for item in items:
        syllable_map[item] = key
NONE_DIACRITIC_SINGLE_VOWELS = set(["a", "e", "i", "o", "u", "y"])
NONE_DIACRITIC_DOUBLE_VOWELS = set([
    "ai", "ao", "au", "ay",
    "eo", "eu", "ia", "ie", "iu", "oa", "oe", "oi", "oo",
    "ua", "ui", "uo", "uu", "uy", "ye"
])
NONE_DIACRITIC_TRIPLE_VOWELS = set([
    "ieu", "oai", "oao", "oay", "oeo",
    "uao", "uay", "uoi", "uou", "uya", "uye", "uyu",
    "yeu"
])
NONE_DIACRITIC_VOWELS = NONE_DIACRITIC_SINGLE_VOWELS | NONE_DIACRITIC_DOUBLE_VOWELS | NONE_DIACRITIC_TRIPLE_VOWELS


class AnalysableWord:
    def __init__(self, word):
        self.word = word
        group = ""
        for syllable in self.word:
            syllable = syllable.lower()
            if syllable in syllable_map:
                normalize_syllable = syllable_map[syllable]
            else:
                normalize_syllable = syllable
            if normalize_syllable in NONE_DIACRITIC_SINGLE_VOWELS:
                group += normalize_syllable
        if group in NONE_DIACRITIC_VOWELS:
            miss_spell = False
        else:
            miss_spell = True
        self.group = group
        self.miss_spell = miss_spell


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

if __name__ == '__main__':
    # write_analyzes()
    count_normalize()
