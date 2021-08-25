from multiprocessing import Pool
from os.path import join, dirname

import requests
from bs4 import BeautifulSoup
from os.path import exists
import joblib

from underthesea.utils import logger

PROJECT_FOLDER = dirname(dirname(dirname(__file__)))
DATASETS_FOLDER = join(PROJECT_FOLDER, "datasets")
DICTIONARY_FOLDER = join(DATASETS_FOLDER, "UD_Vietnamese-COL", "dictionary")


class Sense:
    def __init__(self, tag, sub_tag, description=''):
        self.tag = tag
        self.sub_tag = sub_tag
        self.description = description


class Word:
    def __init__(self, word, senses=[]):
        self.word = word
        self.senses = senses

    def __str__(self):
        content = self.word + " "
        for sense in self.senses:
            content += sense.tag + ","
        return content

    def add_sense(self, sense):
        self.senses.append(sense)

    def has_tag(self, tag) -> bool:
        for sense in self.senses:
            if sense.tag == tag:
                return True
        return False


class Cache:
    def __init__(self, data):
        self.data = data

    def add(self, key, value):
        self.data[key] = value

    def save(self, file):
        joblib.dump(self.data, file)

    def contains(self, key):
        return key in self.data

    def get(self, key):
        if key in self.data:
            return self.data[key]
        return None

    @staticmethod
    def load(file):
        if not exists(file):
            data = {}
            return Cache(data)
        data = joblib.load(file)
        return Cache(data)


class VLSPDictionary:
    def __init__(self):
        pass

    @staticmethod
    def lookups(keywords, n_workers=None):
        with Pool(n_workers) as p:
            data = p.map(VLSPDictionary.lookup, keywords)
        return data

    @staticmethod
    def lookup(keyword, cache=None):
        if cache:
            if cache.contains(keyword):
                return cache.get(keyword)
        logger.info('request ' + keyword)
        url = "https://vlsp.hpda.vn/demo/?page=vcl"
        payload = {"word": keyword}
        r = requests.post(url, data=payload)
        soup = BeautifulSoup(r.content, "html.parser")
        senses_data = soup.select("#vcl_content table .sense")
        if len(senses_data) == 0:
            if cache:
                cache.add(keyword, None)
            return None

        w = Word(keyword, senses=[])
        for sense_data in senses_data:
            tags = sense_data.select(".word_description li")
            # morpho_tag = tags[0]
            syntax_tag = tags[1]
            semantic_tag = tags[2]
            syntax_tag = syntax_tag.select("font")
            tag, sub_tag = syntax_tag[0].text, syntax_tag[1].text
            description = " / ".join([item.text for item in semantic_tag.select("font")])
            sense = Sense(tag, sub_tag, description)
            w.add_sense(sense)
            if cache:
                cache.add(keyword, w)
        return w


if __name__ == '__main__':
    vlsp_cache_file = join(DICTIONARY_FOLDER, "data", "vlsp_cache.bin")
    vlsp_cache = Cache.load(vlsp_cache_file)
    word = VLSPDictionary.lookup("tàu", vlsp_cache)
    print(word)

    # words = [
    #     "đông đúc",
    #     "đậm",
    #     "tiêu biểu",
    #     "cận",
    #     "hữu ích",
    #     "vô hạn",
    #     "rẻ"
    # ]
    # for word in words:
    #     word = VLSPDictionary.lookup(word, vlsp_cache)
    #     print(word)
    #
    # vlsp_cache.save(vlsp_cache_file)
