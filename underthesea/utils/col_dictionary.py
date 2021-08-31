from os import listdir
from os.path import dirname, join
import pandas as pd
import yaml
from underthesea.file_utils import UNDERTHESEA_FOLDER
from underthesea.utils.col_external_dictionary import Cache, VLSPDictionary
from underthesea.utils.col_script import UDDataset

PROJECT_FOLDER = dirname(dirname(dirname(__file__)))
DATASETS_FOLDER = join(PROJECT_FOLDER, "datasets")
COL_FOLDER = join(DATASETS_FOLDER, "UD_Vietnamese-COL")
DICTIONARY_FOLDER = join(DATASETS_FOLDER, "UD_Vietnamese-COL", "dictionary")
DICTIONARY_FILE = join(DICTIONARY_FOLDER, "202108.yaml")


class Word:
    pass


class Dictionary:
    def __init__(self, data):
        self.data = data

    def describe(self):
        print("Dictionary Describe")
        print(f"- Words: {len(self.data)}")

    def import_words(self, data_file):
        df = pd.read_excel(data_file, engine='openpyxl', nrows=5000)
        df = df[df['verify'] == 'x']
        for key, row in df.iterrows():
            token = row[0]
            pos = row[1]
            if token in self.data:
                senses = self.data[token]
                exist_pos = [s['tag'] for s in senses]
                if pos not in exist_pos:
                    self.data[token].append({'tag': pos})
            else:
                self.data[token] = [{'tag': pos}]

    def to_df(self):
        output = []
        for word in self.data:
            for sense in self.data[word]:
                item = {'token': word, 'pos': sense['tag']}
                output.append(item)
        df = pd.DataFrame(output)
        return df

    def save(self, dictionary_file):
        content = yaml.dump(self.data, allow_unicode=True, sort_keys=True)
        with open(dictionary_file, 'w') as f:
            f.write(content)

    @staticmethod
    def load(dictionary_file, cache_file=None):
        with open(DICTIONARY_FILE) as f:
            content = f.read()
        data = yaml.safe_load(content)
        return Dictionary(data)


if __name__ == '__main__':
    dictionary = Dictionary.load(DICTIONARY_FILE)
    dictionary.describe()
    current_df = dictionary.to_df()

    # ud_file = join(COL_FOLDER, "corpus", "ud", "202108.txt")
    ud_folder = join(UNDERTHESEA_FOLDER, "data", "viwiki-20210720", "ud", "AA")
    sentences = []
    for file in listdir(ud_folder)[:2]:
        ud_file = join(UNDERTHESEA_FOLDER, "data", "viwiki-20210720", "ud", "AA", file)
        ud_dataset = UDDataset.load(ud_file)
        sentences += ud_dataset.sentences
    ud_dataset = UDDataset(sentences)
    rows = [s.rows for s in ud_dataset]
    rows = [[row[0].lower(), row[1]] for sublist in rows for row in sublist]
    ud_df = pd.DataFrame(rows, columns=['token', 'pos'])
    print("List pos:", sorted(set(ud_df["pos"])))
    pos = [
        ["V", "verb"],
        ["N", "noun"],
        ["A", "adjective"],
        ["P", "pronoun"],
        ["E", "preposition"]
    ]
    vlsp_cache_file = join(DICTIONARY_FOLDER, "data", "vlsp_cache.bin")
    vlsp_cache = Cache.load(vlsp_cache_file)

    for pos_label, pos_full_name in pos:
        ud_df_sub = ud_df[ud_df["pos"].isin([pos_label])]
        ud_df_sub = ud_df_sub.groupby(["token", "pos"]).size().reset_index(name='count').sort_values("count",
                                                                                                     ascending=False)

        df = pd.merge(ud_df_sub, current_df, on=['token', 'pos'], how='outer', indicator=True)
        df = df[df['_merge'] == 'left_only']
        df = df[['token', 'pos', 'count']]
        df['verify'] = ''
        all_tokens = [item['token'] for i, item in df.iterrows()]

        tokens = [token for token in all_tokens if not vlsp_cache.contains(token)]
        MAX_ITEMS = 500
        words_data = VLSPDictionary.lookups(tokens[:MAX_ITEMS], n_workers=30)
        for token, word in zip(tokens, words_data):
            vlsp_cache.add(token, word)

        words = [vlsp_cache.get(token) for token in all_tokens]

        def check_has_tag(data):
            word, items = data
            tag = items[1]['pos']
            if word is None:
                return '', ''
            for sense in word.senses:
                if sense.tag == tag:
                    return 'x', sense.description
            return '', ''

        vlsp_data = [check_has_tag(data) for data in zip(words, df.iterrows())]
        votes, descriptions = zip(*vlsp_data)
        df['vlsp_votes'] = votes
        df['vlsp_description'] = descriptions
        file = join(DICTIONARY_FOLDER, "data", f"words_{pos_full_name}_candidates.xlsx")
        df.to_excel(file, index=False)
    vlsp_cache.save(vlsp_cache_file)
