import os
from os.path import join, dirname
from elasticsearch import Elasticsearch

from underthesea.utils.col_dictionary import Dictionary

PROJECT_FOLDER = dirname(dirname(__file__))
os.sys.path.append(PROJECT_FOLDER)

DICTIONARY_FILE = join(PROJECT_FOLDER, "datasets", "dictionary", "202108.yaml")
dictionary = Dictionary.load(dictionary_file=DICTIONARY_FILE)


def index_dictionary(dictionary, es, index_name):
    for item in dictionary.data:
        print('Index ', item)
        try:
            item_data = dictionary.data[item]
            doc = {
                'word': item,
                'data': item_data
            }
            es.index(index=index_name, body=doc)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    es = Elasticsearch()
    # index_dictionary(dictionary, es, index_name="dictionary")
