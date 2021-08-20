import math
from underthesea.dictionary import Dictionary
from collections import Counter, defaultdict
from underthesea.utils import logger
from datetime import date, timedelta
import pandas as pd


def get_words_pos(dictionary):
    output = {}
    for word in dictionary.words:
        data = dictionary.words[word]
        pos = list(set([_['pos'] for _ in data]))
        output[word] = pos
    return output


dictionary = Dictionary.Instance()
n_words = len(dictionary.words)
print(f"Load dictionary with {n_words} words.")
words_pos = get_words_pos(dictionary)


def computeIDF(docCounters):
    """Get Inverse Document Frequency value for each word"""
    n = len(docCounters)
    idfDict = defaultdict(int)
    for doc in docCounters:
        for word, freq in doc.items():
            if freq > 0:
                idfDict[word] += 1

    for word, val in idfDict.items():
        idfDict[word] = math.log10(n / float(val))
    return idfDict


class UDAnalyzer:
    def __init__(self):
        self.total_errors = 0

    def _get_words(self, sentences):
        tags = [s.rows for s in sentences]
        tags = [t for sublist in tags for t in sublist]
        words = [t[0].lower() for t in tags]
        return words

    def _get_doc_sents(self, dataset):
        """Return doc as key and doc sentence array as value"""
        doc_sents = defaultdict(list)
        for s in dataset:
            doc = s.headers['doc_url']
            doc_sents[doc].append(s)
        return doc_sents

    def get_doc_word_counters(self, dataset):
        """Get word count by doc url"""
        data = self._get_doc_sents(dataset)
        doc_word_counters = {}
        for doc, sents in data.items():
            tags = [s.rows for s in sents]
            tags = [t for sublist in tags for t in sublist]
            words = [t[0].lower() for t in tags]
            doc_word_counters[doc] = Counter(words)
        return doc_word_counters

    def analyze_words_pos(self, sentences):
        tags = [s.rows for s in sentences]
        tags = [t for sublist in tags for t in sublist]
        df = pd.DataFrame(tags, columns=["word", "pos", "order", "dep"])
        return df

    def analyze_words(self, sentences):
        tags = [s.rows for s in sentences]
        tags = [t for sublist in tags for t in sublist]
        words = [t[0].lower() for t in tags if t[1] not in ["Np", "CH"]]
        noun_phrases = [t[0] for t in tags if t[1] == "Np"]
        counter = Counter(words)
        corpus_words = set(counter.keys())
        dictionary_words = set(dictionary.words)
        oov = corpus_words - dictionary_words

        print("Most common words:\n", counter.most_common(50))
        print("Corpus words: ", len(corpus_words))
        print("Noun phrases:\n", set(noun_phrases))
        print("Short OOV words:\n", sorted([item for item in oov if len(item.split(" ")) < 3]))
        print("Long OOV words:\n", sorted([item for item in oov if len(item.split(" ")) >= 3]))
        return counter

    def analyze_all_words(self, dataset):
        return self.analyze_words(dataset)

    def get_today_sentences(self, dataset):
        yesterday = (date.today() - timedelta(days=1)).strftime('%Y%m%d')
        sentences = [s for s in dataset if s.headers['date'] == yesterday]
        return sentences

    def analyze_today_words(self, dataset):
        sentences = self.get_today_sentences(dataset)
        return self.analyze_words(sentences)

    def analyze_sent_ids(self, dataset):
        """Get sent_id of all sentences"""
        sent_ids = [s.headers['sent_id'] for s in dataset]
        counter = Counter(sent_ids)

        duplicate_ids = [key for key in counter if counter[key] > 1]
        if len(duplicate_ids) > 0:
            logger.error('[ERROR] duplicate_ids' + str(duplicate_ids))
        else:
            logger.debug("sent_ids is valid.")

        missing_ids = [str(i + 1) for i in range(len(sent_ids)) if str(i + 1) not in sent_ids]
        if len(missing_ids) > 0:
            logger.error('[ERROR] missing_ids' + str(missing_ids))
        return sent_ids

    def analyze_doc_sent_freq(self, dataset):
        """Get sentence count by doc url"""
        data = self._get_doc_sents(dataset)
        print("Number of doc URLs %s" % len(data))
        doc_sent_counts = [(doc, len(sents)) for doc, sents in data.items()]
        return doc_sent_counts

    def analyze_dataset_len(self, dataset):
        print("Number of sentences", len(dataset))

    def analyze(self, dataset):
        self.analyze_dataset_len(dataset)
        self.analyze_sent_ids(dataset)
        self.analyze_words(dataset)
        self.analyze_doc_sent_freq(dataset)
        self.get_doc_word_counters(dataset)
