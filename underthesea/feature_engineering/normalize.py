from datetime import datetime
from os.path import join
from viet_text_tools import normalize_diacritics
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


# nguyên âm: a, ă, â, e, ê, i, o, ô, ơ, u, ư, y
# phụ âm: b, c, d, đ, g, h, k, l, m, n, p, q, r, s, t, v, x
# phụ âm ghép: ch, gh, gi, kh, ng, ngh, nh, ph, th, tr, qu,
base_norm_dict = {
    # ================================
    # QU or GI
    # ================================
    "qủa": "quả", "qúa": "quá",
    "qúi": "quí",
    "gìa": "già", "gía": "giá",
    "gíá": "giá",
    "gíáo": "giáo",
    "giaó": "giáo",
    "gíang": "giáng",
    # ================================
    # AI
    # ================================
    "aì": "ài",
    "daì": "dài", "đaì": "đài", "haì": "hài", "taì": "tài", "vaì": "vài",
    "ngaì": "ngài",
    "aỉ": "ải",
    "haỉ": "hải", "taỉ": "tải",
    "nhaỉ": "nhải", "phaỉ": "phải", "traỉ": "trải",
    "aĩ": "ãi",
    "baĩ": "bãi",
    "aí": "ái",
    "caí": "cái", "gaí": "gái", "maí": "mái",
    "phaí": "phái", "thaí": "thái", "traí": "trái",
    "aị": "ại",
    "điạ": "địa", "haị": "hại", "laị": "lại", "maị": "mại", "saị": "sại", "taị": "tại",
    "khaị": "khại", "traị": "trại",
    "vaị": "vại", "đaị": "đại",
    # ================================
    # AO
    # ================================
    "aò": "ào",
    "vaò": "vào",
    "aỏ": "ảo",
    "saỏ": "sảo",
    "aó": "áo",
    # ================================
    # AU
    # ================================
    "saụ": "sạu",
    "nhaụ": "nhạu",
    "đâù": "đầu",
    "đâụ": "đậu",
    # ================================
    # AY
    # ================================
    "aỵ": "ạy",
    "haỵ": "hạy", "naỵ": "nạy", "taỵ": "tạy",
    "ngaỵ": "ngạy",
    "đâỵ": "đậy",
    # ================================
    # IA
    # ================================
    "baì": "bài", "kià": "kìa",
    "iá": "ía",
    "tiá": "tía",
    "phiá": "phía",
    "iã": "ĩa",
    "nghiã": "nghĩa",
    # ================================
    # IEU
    # ================================
    "đìều": "điều",
    "nhiêù": "nhiều",
    # ================================
    # OA
    # ================================
    "oà": "òa",
    "đoà": "đòa", "hoà": "hòa", "goà": "gòa", "lòa": "loà", "nòa": "noà", "toà": "tòa", "voà": "vòa", "xoà": "xòa",
    "ngoà": "ngòa", "nhoà": "nhòa",
    "oá": "óa",
    "đoá": "đóa", "hoá": "hóa", "goá": "góa", "loá": "lóa", "toá": "tóa", "xoá": "xóa",
    "khoá": "khóa", "thoá": "thóa",
    "oả": "ỏa",
    "hoả": "hỏa", "loả": "lỏa", "toả": "tỏa", "soả": "sỏa", "xoả": "xỏa",
    "khoả": "khỏa", "thoả": "thỏa", "troả": "trỏa",
    "oã": "õa",
    "hoã": "hõa", "loã": "lõa", "toã": "tõa", "xoã": "xõa",
    "thoã": "thõa", "ngoã": "ngõa",
    "doạ": "dọa", "đoạ": "đọa", "hoạ": "họa", "loạ": "lọa", "toạ": "tọa", "xoạ": "xọa",
    "khoạ": "khọa", "ngoạ": "ngọa", "thoạ": "thọa",
    # ================================
    # OAC - OACH
    # ================================
    "họach": "hoạch",
    # ================================
    # OAT
    # ================================
    "khóat": "khoát",
    # ================================
    # OE
    # ================================
    "oè": "òe",
    "hoè": "hòe", "loè": "lòe", "toè": "tòe", "xoè": "xòe",
    "choè": "chòe", "khoè": "khòe", "ngoè": "ngòe", "nhoè": "nhòe",
    "oé": "óe",
    "loé": "lóe", "toé": "tóe", "voé": "vóe", "xoé": "xóe",
    "choé": "chóe", "khoé": "khóe", "ngoé": "ngóe", "nghoé": "nghóe", "nhoé": "nhóe", "phoé": "phóe",
    "oẻ": "ỏe",
    "hoẻ": "hỏe", "loẻ": "lỏe", "toẻ": "tỏe",
    "khoẻ": "khỏe", "nhoẻ": "nhỏe",
    "oẽ": "õe",
    "loẽ": "lõe",
    "khoẽ": "khõe",
    "oẹ": "ọe",
    "hoẹ": "họe", "toẹ": "tọe", "xoẹ": "xọe",
    "choẹ": "chọe", "khoẹ": "khọe",
    # ================================
    # OI
    # ================================
    "rôì": "rồi",
    "ôị": "ội",
    "tôị": "tội",
    "thôị": "thội",
    # ================================
    # UA
    # ================================
    "muà": "mùa",
    "chuà": "chùa",
    "uá": "úa",
    "chuá": "chúa",
    "uả": "ủa",
    "buả": "bủa", "cuả": "của", "đuả": "đủa", "huả": "hủa", "ruả": "rủa", "suả": "sủa", "vuả": "vủa", "xuả": "xủa",
    "chuả": "chủa", "khuả": "khủa",
    "nưã": "nữa",
    # ================================
    # UY
    # ================================
    "uỳ": "ùy",
    "qùy": "quỳ",  # special case ?
    "tuỳ": "tùy", "xuỳ": "xùy",
    "chuỳ": "chùy", "thuỳ": "thùy", "truỳ": "trùy",
    "uý": "úy",
    "qúy": "quý",  # special case ?
    "huý": "húy", "luý": "lúy", "suý": "súy", "tuý": "túy", "xuý": "xúy",
    "thuý": "thúy",
    "uỷ": "ủy",
    "huỷ": "hủy", "tuỷ": "tủy",
    "qủy": "quỷ",  # special case ?
    "chuỷ": "chủy", "khuỷ": "khủy", "thuỷ": "thủy", "truỷ": "trủy",
    "uỹ": "ũy",
    "qũy": "quỹ",  # special case ?
    "luỹ": "lũy",
    "thuỹ": "thũy",
    "uỵ": "ụy",
    "qụy": "quỵ",  # special case ?
    "huỵ": "hụy", "luỵ": "lụy", "tuỵ": "tụy",
    "khuỵ": "khụy", "nguỵ": "ngụy", "nhuỵ": "nhụy", "thuỵ": "thụy", "truỵ": "trụy",
    # ================================
    # UI
    # ================================
    "suỉ": "sủi",
    # ================================
    # OAI
    # ================================
    "òai": "oài",
    "gòai": "goài", "hòai": "hoài", "lòai": "loài",
    "ngòai": "ngoài", "nhòai": "nhoài",
    "oaì": "oài",
    "ngoaì": "ngoài",
    "óai": "oái",
    "khóai": "khoái",
    "ọai": "oại",
    "lọai": "loại",
    "ngọai": "ngoại", "thọai": "thoại",
    # ================================
    # OAN - OANG - OANH
    # ================================
    "òan": "oàn",
    "đòan": "đoàn", "hòan": "hoàn", "tòan": "toàn",
    "óan": "oán",
    "đóan": "đoán", "tóan": "toán",
    "khỏan": "khoản",
    "ọan": "oạn",
    "đọan": "đoạn", "lọan": "loạn",
    "òang": "oàng",
    "chòang": "choàng", "hòang": "hoàng", "tòang": "toàng", "xòang": "xoàng",
    "óang": "oáng",
    "thóang": "thoáng",
    "ỏang": "oảng",
    "hỏang": "hoảng",
    "khỏang": "khoảng",
    "òanh": "oành",
    "hòanh": "hoành",
    # ================================
    # OAT
    # ================================
    "óat": "oát",
    "thóat": "thoát",
    "ọat": "oạt",
    "đọat": "đoạt",
    "họat": "hoạt",
    # ================================
    # UOI
    # ================================
    "ngừơi": "người",
    "ngươì": "người",
    # ================================
    # UOC
    # ================================
    "úơc": "ước",
    "bưóc": "bước",
    "trưóc": "trước",
    "trứơc": "trước",
    "trưóc": "trước",
    "chưóc": "chước",
    "đưọc": "được",
    "đựơc": "được",
    "đựợc": "đượ̣c",
    # ================================
    # UON - UONG
    # ================================
    "trửơng": "trưởng",
    # ================================
    # UYEN
    # ================================
    "chuỷên": "chuyển",
    # ================================
    # UYU
    # ================================
    "khủyu": "khuỷu"
}
# load base_norm_dict from rules.xlsx file
base_norm_dict = {}
rules_df = pd.read_excel("rules.xlsx")
for key, item in rules_df.iterrows():
    word = item[0]
    normalize = item[1]
    base_norm_dict[word] = normalize
norm_dict = base_norm_dict.copy()
# add capitalize rules
for key in base_norm_dict:
    new_key = key.capitalize()
    new_value = base_norm_dict[key].capitalize()
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
    "a": "ảáăâàấẵầằáắậââạạăãảạẩâ",
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


def compare_with_lab_viet_text_tools():
    n_diff = 0
    ignores = set([
        "loà", "đưọc", "Gassée"
    ])
    df = pd.DataFrame(columns=["word", "lower", "vtt", "uts", "group", "miss_spell"])
    with open(TOKENS_ANALYSE_FILE) as f:
        data = {}
        for line in f:
            word, freq = line.split("\t\t")
            vtt_words = normalize_diacritics(word)
            uts_words = normalize(word)
            if word != "nghiêng" and len(word) > 6:
                continue
            if word in ignores:
                continue
            if vtt_words != word and vtt_words != uts_words:
                analysable_word = AnalysableWord(word)
                data[analysable_word] = {
                    "vtt": vtt_words,
                    "uts": uts_words
                }
                item = pd.DataFrame([{
                    "word": analysable_word.word,
                    "lower": analysable_word.word.lower(),
                    "group": analysable_word.group,
                    "miss_spell": analysable_word.miss_spell,
                    "vtt": vtt_words,
                    "uts": uts_words
                }])
                df = pd.concat([df, item], ignore_index=True)
                n_diff += 1
        df = df.sort_values(by=["miss_spell", "group", "lower"], ascending=True)
        df.to_excel("data.xlsx", index=False)
        for analysable_word in data:
            item = data[analysable_word]
            print(f"{analysable_word.word} -> {item['vtt']} | {item['uts']}")
    print(f"Differences: {n_diff}")


if __name__ == '__main__':
    # write_analyzes()
    # count_normalize()
    compare_with_lab_viet_text_tools()
