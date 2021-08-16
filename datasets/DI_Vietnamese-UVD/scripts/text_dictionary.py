import json
import os
from os.path import join, exists


class Word:
    def __init__(self, text):
        self.text = text

    def set_source(self, source):
        self.source = source


class Dictionary:
    def __init__(self, words=None):
        self.words = words

    def save(self, dictionary_path):
        file = join(dictionary_path, "words.txt")
        if exists(file):
            os.remove(file)
        f = open(file, "a")
        for word in self.words:
            content = {"text": word.text, "source": word.source}
            content = json.dumps(content, ensure_ascii=False)
            f.write(content + "\n")

    def load(self, dictionary_path):
        self.words = []
        for line in open(join(dictionary_path, "words.txt")):
            content = line.strip()
            item = json.loads(content)
            word = Word(item["text"])
            word.set_source(item["source"])
            self.words.append(word)


class DictionaryUtil:
    @staticmethod
    def merge(dictionaries):
        data = {}
        for dictionary in dictionaries:
            for word in dictionary.words:
                text = word.text
                if text not in data:
                    data[text] = {
                        "source": [word.source]
                    }
                else:
                    sources = data[text]["source"]
                    sources.append(word.source)
                    sources = list(sorted(set(sources)))
                    data[text]["source"] = sources
        words = []
        for text in data:
            word = Word(text)
            word.source = data[text]["source"]
            words.append(word)
        dictionary = Dictionary(words)
        return dictionary
