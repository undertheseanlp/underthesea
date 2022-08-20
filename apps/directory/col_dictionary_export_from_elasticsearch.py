from elasticsearch import Elasticsearch
from apps.directory.utils import DICTIONARY_FILE
import yaml


def query_doc(es, index_name, scroll_time="1s"):
    body = {
        "sort": [{
            "_id": {"order": "asc"}
        }],
        "query": {"match_all": {}},
        "size": 2000
    }
    res = es.search(index=index_name, body=body, scroll=scroll_time)
    yield res
    scroll_id = res["_scroll_id"]
    while len(res["hits"]["hits"]):
        res = es.scroll(
            scroll_id=scroll_id,
            scroll=scroll_time
        )
        scroll_id = res["_scroll_id"]
        yield res


def extract_dictionary_from_elasticsearch(data):
    dict_data = {}
    for item in data:
        headword = item["_id"]
        sense_data = item["_source"]["senses"]
        dict_data[headword] = sense_data
    return dict_data


if __name__ == '__main__':
    es = Elasticsearch()
    index_name = "dictionary"
    with open(DICTIONARY_FILE, "w") as f:
        f.write("")
    f = open(DICTIONARY_FILE, "a")
    i = 0
    for res in query_doc(es, index_name):
        docs = res["hits"]["hits"]
        print(len(docs))
        doc_content = extract_dictionary_from_elasticsearch(docs)
        if len(doc_content) > 1:
            dict_content = yaml.dump(doc_content, allow_unicode=True, sort_keys=True)
            f.write(dict_content)
            i += 1
    f.close()
