import pandas as pd
import yaml

from examples.dictionary.col_dictionary import DICTIONARY_FILE


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
