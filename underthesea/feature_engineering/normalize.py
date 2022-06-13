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


def normalize(s):
    # b, c, d, đ, g, h, k, l, m, n, p, q, r, s, t, v, x
    # phụ âm ghép: ch, gh, gi, kh, ng, ngh, nh, ph, th, tr, qu,
    norm_dict = {
        "oà": "òa",
        "đoà": "đòa", "hoà": "hòa", "goà": "gòa", "lòa": "loà", "nòa": "noà", "toà": "tòa", "voà": "vòa", "xoà": "xòa",
        "Đoà": "Đòa", "Hoà": "Hòa", "Goà": "Gòa", "Lòa": "Loà", "Nòa": "Noà", "Toà": "Tòa", "Voà": "Vòa", "Xoà": "Xòa",
        "ngoà": "ngòa", "nhoà": "nhòa",
        "Ngoà": "Ngòa", "Nhoà": "Nhòa",
        "oá": "óa",
        "đoá": "đóa", "hoá": "hóa", "goá": "góa", "loá": "lóa", "toá": "tóa", "xoá": "xóa",
        "Đoá": "Đóa", "Hoá": "Hóa", "Goá": "Góa", "Loá": "Lóa", "Toá": "Tóa", "Xoá": "Xóa",
        "khoá": "khóa", "thoá": "thóa",
        "Khoá": "Khóa", "Thoá": "Thóa",
        "oả": "ỏa",
        "hoả": "hỏa", "loả": "lỏa", "toả": "tỏa", "soả": "sỏa", "xoả": "xỏa",
        "Hoả": "Hỏa", "Loả": "Lỏa", "Toả": "Tỏa", "Soả": "Sỏa", "Xoả": "Xỏa",
        "khoả": "khỏa", "thoả": "thỏa", "troả": "trỏa",
        "Khoả": "Khỏa", "Thoả": "Thỏa", "Troả": "Trỏa",
        "oã": "õa",
        "hoã": "hõa", "loã": "lõa", "toã": "tõa", "xoã": "xõa",
        "Hoã": "Hõa", "Loã": "Lõa", "Toã": "Tõa", "Xoã": "Xõa",
        "thoã": "thõa", "ngoã": "ngõa",
        "Thoã": "Thõa", "Ngoã": "Ngõa",
        "doạ": "dọa", "đoạ": "đọa", "hoạ": "họa", "loạ": "lọa", "toạ": "tọa", "xoạ": "xọa",
        "Doạ": "Dọa", "Đoạ": "Đọa", "Hoạ": "Họa", "Loạ": "Lọa", "Toạ": "Tọa", "Xoạ": "Xọa",
        "khoạ": "khọa", "ngoạ": "ngọa", "thoạ": "thọa",
        "Khoạ": "Khọa", "Ngoạ": "Ngọa", "Thoạ": "Thọa",
        "oè": "òe",
        "hoè": "hòe", "loè": "lòe", "toè": "tòe", "xoè": "xòe",
        "Hoè": "Hòe", "Loè": "Lòe", "Toè": "Tòe", "Xoè": "Xòe",
        "choè": "chòe", "khoè": "khòe", "ngoè": "ngòe", "nhoè": "nhòe",
        "Choè": "Chòe", "Khoè": "Khòe", "Ngoè": "Ngòe", "Nhoè": "Nhòe",
        "oé": "óe",
        "Loé": "Lóe", "Toé": "Tóe", "Voé": "Vóe", "Xoé": "Xóe",
        "choé": "chóe", "khoé": "khóe", "ngoé": "ngóe", "nghoé": "nghóe", "nhoé": "nhóe", "phoé": "phóe",
        "Choé": "Chóe", "Khoé": "Khóe", "Ngoé": "Ngóe", "Nghoé": "Nghóe", "Nhoé": "Nhóe", "Phoé": "Phóe",
        "suỉ": "sủi"
    }
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


if __name__ == '__main__':
    # write_analyzes()
    count_normalize()
