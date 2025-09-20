# Script for evaluation Vietnamese normalizers
from os.path import join
import pandas as pd
# from tools import vtm
from tools import vtt
# from tools import nvh
from examples.text_normalize.normalize import CORPUS_FOLDER, AnalysableWord
from underthesea import text_normalize

TOKENS_ANALYSE_FILE = join(CORPUS_FOLDER, "tokens_analyze.txt")

# Normalizer = vtm
Normalizer = vtt
# Normalizer = nvh


def compare_two_tools():
    df = pd.DataFrame(columns=["word", "lower", "other", "uts", "group", "miss_spell"])
    with open(TOKENS_ANALYSE_FILE) as f:
        data = {}
        for i, line in enumerate(f):
            word, freq = line.split("\t\t")
            other_words = Normalizer.normalize(word)
            uts_words = text_normalize(word)
            if word != "nghiêng" and len(word) >= 7:
                continue
            if other_words != word and other_words != uts_words:
                analysable_word = AnalysableWord(word)
                data[analysable_word] = {
                    "other": other_words,
                    "uts": uts_words
                }
                item = pd.DataFrame([{
                    "word": analysable_word.word,
                    "lower": analysable_word.word.lower(),
                    "group": analysable_word.group,
                    "miss_spell": analysable_word.miss_spell,
                    "other": other_words,
                    "uts": uts_words
                }])
                df = pd.concat([df, item], ignore_index=True)
        df = df.sort_values(by=["miss_spell", "group", "lower"], ascending=True)
        df.to_excel("tmp/results.xlsx", index=False)
    total = df.shape[0]
    non_miss_spell = df[df["miss_spell"] == 0].shape[0]
    miss_spell = df[df["miss_spell"] == 1].shape[0]
    print(f"Total differences: {total} (non_miss_spell: {non_miss_spell}, miss_spell: {miss_spell})")


if __name__ == '__main__':
    compare_two_tools()
