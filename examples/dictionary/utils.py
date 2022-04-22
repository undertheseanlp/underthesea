from os.path import dirname, join

import pandas as pd
import yaml

PROJECT_FOLDER = dirname(dirname(dirname(__file__)))
DATASETS_FOLDER = join(PROJECT_FOLDER, "datasets")
COL_FOLDER = join(DATASETS_FOLDER, "UD_Vietnamese-COL")
DICTIONARY_FOLDER = join(DATASETS_FOLDER, "UD_Vietnamese-COL", "dictionary")
DICTIONARY_FILE = join(DICTIONARY_FOLDER, "202108.yaml")


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


class DictionarySense:
    def __init__(self):
        pass


class DictionaryWord:
    def __init__(self, text, senses=[]):
        self.text = text
        self.senses = senses


class CoreDictionary:
    def __init__(self):
        self.words = {}

    def add_word(self, word):
        self.words[word] = word

    def to_yaml(self, save_file):
        content = ""
        i = 0
        for word in self.words:
            i += 1
            if i > 100:
                continue
            content += f"{word.text}:\n"
            for sense in word.senses:
                content += f"- description: {sense['definition']}\n"
                content += f"  examples:\n"
                if sense['example']:
                    content += f"  - {sense['example']}\n"
                content += f"  tag: {sense['pos']}\n"
        with open(save_file, "w") as f:
            f.write(content)

    @staticmethod
    def load_yaml(save_file):
        with open(save_file) as f:
            data = yaml.safe_load(f)
        dictionary = CoreDictionary()
        for text in data:
            word = DictionaryWord(text, senses=[])
            dictionary.add_word(word)
        return dictionary
