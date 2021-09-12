from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search


class Dictionary:
    def __init__(self, es_index='dictionary'):
        self.es_index = es_index
        self.es = Elasticsearch()
        s = Search(using=self.es, index=es_index)
        ids = [h.meta.id for h in s.scan()]
        words = sorted(ids)
        self.words = words

    def get_word(self, word):
        if word not in self.words:
            return {}
        print(word)
        body = self.es.get(index=self.es_index, id=word)
        data = body["_source"]
        return data

    def save(self, word, data):
        self.es.index(index=self.es_index, id=word, body=data)


if __name__ == '__main__':
    dictionary = Dictionary(es_index='dictionary')
