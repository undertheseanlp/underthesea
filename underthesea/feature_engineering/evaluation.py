# Script for evaluation Vietnamese normalizers
from os.path import join
import pandas as pd
from tools import vtm

from underthesea.feature_engineering.normalize import CORPUS_FOLDER, normalize, AnalysableWord

TOKENS_ANALYSE_FILE = join(CORPUS_FOLDER, "tokens_analyze.txt")

Normalizer = vtm


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
            vtt_words = Normalizer.normalize(word)
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
        df.to_excel("results.xlsx", index=False)
    total = df.shape[0]
    non_miss_spell = df[df["miss_spell"] == 0].shape[0]
    miss_spell = df[df["miss_spell"] == 1].shape[0]
    print(f"Total differences: {total} (non_miss_spell: {non_miss_spell}, miss_spell: {miss_spell})")


if __name__ == '__main__':
    compare_with_lab_viet_text_tools()
