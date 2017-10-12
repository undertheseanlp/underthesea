from collections import Counter

import pandas as pd

from analyze_characters import get_utf8_number, get_unicode_number
from underthesea.corpus import PlainTextCorpus
from vietnamese_normalize import vietnamese_normalize

corpus = PlainTextCorpus()
# corpus_dir = "D:\\PycharmProjects\\underthesea\\corpus.vinews\\vn_news\\data"
corpus_dir = "D:\\PycharmProjects\\_NLP_DATA\\vlsp 2016\\sa\\SA2016-training_data"
corpus_dir = "D:\\PycharmProjects\\1link\\opinion_mining\\data\\foody"

corpus.load(corpus_dir)

sentences = sum([d.sentences for d in corpus.documents], [])
text = u" ".join(sentences[:200])
text = vietnamese_normalize(text)
counter = Counter(text)
df = pd.DataFrame.from_dict(counter, orient='index').reset_index()
df.columns = ["character", "freq"]
df["unicode"] = df.apply(lambda row: get_unicode_number(row["character"]), axis=1)
df["utf-8"] = df.apply(lambda row: get_utf8_number(row["character"]), axis=1)
df = df.sort_values(["freq"], ascending=False)
df.to_excel("analyze.xlsx", index=False)
corpus_character_sets = set(df["character"])


def load_known_characters():
    files = ["tcvn_6909_2001.xlsx", "other_characters.xlsx"]
    df = pd.concat([pd.read_excel(f) for f in files])
    characters = set(df["character"])
    return characters


known_characters = load_known_characters()
new_characters = list(corpus_character_sets - known_characters)

new_characters_df = pd.DataFrame({"character": new_characters})
new_characters_df["unicode"] = new_characters_df.apply(lambda row: get_unicode_number(row["character"]), axis=1)
new_characters_df = new_characters_df.sort_values(["unicode"], ascending=False)
new_characters_df.to_excel("new_characters.xlsx", index=False)

print(list(new_characters))
for c in new_characters:
    print c
