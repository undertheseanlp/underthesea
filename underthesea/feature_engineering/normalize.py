from datetime import datetime
from os.path import join

from underthesea.file_utils import UNDERTHESEA_FOLDER
from underthesea.pipeline.word_tokenize import tokenize

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
    "aỉ": "ải",
    "aị": "ại",
    "điạ": "địa", "laị": "lại",
    # ================================
    # AO
    # ================================
    "vaò": "vào",
    "aỏ": "ảo",
    "saỏ": "sảo",
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
    "trưóc": "trước",
    "đưọc": "được",
    "đựơc": "được",
    "đựợc": "đượ̣c",
    "trứơc": "trước",
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
    filename = join(CORPUS_FOLDER, 'token_analyze.txt')
    total_count = 0
    normalized_count = 0
    with open(filename) as f:
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


from viet_text_tools import normalize_diacritics


def compare_with_lab_viet_text_tools():
    filename = join(CORPUS_FOLDER, 'token_analyze.txt')
    n_diff = 0
    ignores = set([
        "loà", "đưọc", "Gassée"
    ])
    with open(filename) as f:
        for line in f:
            word, freq = line.split("\t\t")
            vtt_words = normalize_diacritics(word)
            uts_words = normalize(word)
            if word != "nghiêng" and len(word) > 6:
                continue
            if word in ignores:
                continue
            if vtt_words != word and vtt_words != uts_words:
                print(f"{word} -> {vtt_words} | {uts_words}")
                n_diff += 1
    print(f"Differents: {n_diff}")


if __name__ == '__main__':
    # write_analyzes()
    # count_normalize()
    compare_with_lab_viet_text_tools()
