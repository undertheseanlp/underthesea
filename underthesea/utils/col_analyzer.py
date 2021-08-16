from underthesea.dictionary import Dictionary
from collections import Counter
from underthesea.utils import logger
from datetime import date, timedelta

dictionary = Dictionary.Instance()
n_words = len(dictionary.words)
print(f"Load dictionary with {n_words} words.")


class UDAnalyzer:
    def __init__(self):
        self.total_errors = 0

    def _get_words(self, sentences):
        tags = [s.tags for s in sentences]
        tags = [t for sublist in tags for t in sublist]
        words = [t[0].lower() for t in tags]
        return words

    def analyze_words(self, sentences):
        tags = [s.tags for s in sentences]
        tags = [t for sublist in tags for t in sublist]
        words = [t[0].lower() for t in tags]
        counter = Counter(words)
        print("Most common words")
        print(counter.most_common(50))
        corpus_words = set(counter.keys())
        print("Corpus words: ", len(corpus_words))
        dictionary_words = set(dictionary.words)
        oov = corpus_words - dictionary_words
        print("OOV words")
        print(oov)
        return counter

    def analyze_all_words(self, dataset):
        return self.analyze_words(dataset)

    def analyze_today_words(self, dataset):
        yesterday = (date.today() - timedelta(days=1)).strftime('%Y%m%d')
        sentences = [s for s in dataset if s.date == yesterday]
        return self.analyze_words(sentences)

    def analyze_sent_ids(self, dataset):
        """Get sent_id of all sentences
        """
        sent_ids = [s.sent_id for s in dataset]
        counter = Counter(sent_ids)

        duplicate_ids = [key for key in counter if counter[key] > 1]
        if len(duplicate_ids) > 0:
            logger.error('[ERROR] duplicate_ids' + str(duplicate_ids))
        else:
            logger.debug("send_ids is valid.")
        return sent_ids

    def analyze_dataset_len(self, dataset):
        print("Number of sentences", len(dataset))

    def analyze(self, dataset):
        self.analyze_dataset_len(dataset)
        self.analyze_sent_ids(dataset)
        self.analyze_words(dataset)
