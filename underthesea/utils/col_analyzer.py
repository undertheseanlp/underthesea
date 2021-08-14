from underthesea.dictionary import Dictionary
from underthesea.utils import logger
from collections import Counter

dictionary = Dictionary.Instance()
n_words = len(dictionary.words)
print(f"Load dictionary with {n_words} words.")


class UDAnalyzer:
    def __init__(self):
        self.total_errors = 0

    def analyze_words(self, dataset):
        tags = [s.tags for s in dataset]
        tags = [t for sublist in tags for t in sublist]
        words = [t[0].lower() for t in tags]
        counter = Counter(words)
        print("Words")
        print(counter.most_common(50))
        print("Out of dictionary")
        corpus_words = set(counter.keys())
        dictionary_words = set(dictionary.words)
        print("Corpus words: ", len(corpus_words))
        oov = corpus_words - dictionary_words
        print("OOV words")
        print(oov)

    def analyze_sent_ids(self, dataset):
        """Get sent_id of all sentences
        """
        [s.sent_id for s in dataset]
        logger.debug("send_ids is valid.")

    def analyze(self, dataset):
        print("Number of sentences", len(dataset))
        self.analyze_sent_ids(dataset)
        self.analyze_words(dataset)


if __name__ == '__main__':
    print(0)
