from os.path import dirname, join
import pandas as pd
import yaml

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

    ud_file = join(COL_FOLDER, "corpus", "ud", "202108.txt")
    ud_dataset = UDDataset.load(ud_file)
    rows = [s.rows for s in ud_dataset]
    rows = [[row[0].lower(), row[1]] for sublist in rows for row in sublist]
    df = pd.DataFrame(rows, columns=['token', 'pos'])
    df = df[df["pos"].isin(["N"])]
    ud_df = df.sort_values(["token", "pos"]).drop_duplicates().sort_values(["token", "pos"])

    df = pd.merge(ud_df, current_df, on=['token', 'pos'], how='outer', indicator=True)
    df = df[df['_merge'] == 'left_only']
    df = df[['token', 'pos']]
    df['verify'] = ''
    df.to_excel(join(DICTIONARY_FOLDER, "words_noun_candidates.xlsx"), index=False)
