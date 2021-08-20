import os
from math import log
from os.path import dirname, join
from collections import defaultdict
from underthesea.utils.col_script import UDDataset
from underthesea.utils.col_analyzer import UDAnalyzer

BOT_VERSION = "underthesea.v1.3.2"
PROJECT_FOLDER = dirname(dirname(dirname((os.getcwd()))))
DATASETS_FOLDER = join(PROJECT_FOLDER, "datasets")
COL_FOLDER = join(DATASETS_FOLDER, "UD_Vietnamese-COL")
STOPWORDS_FOLDER = join(PROJECT_FOLDER, "underthesea", "datasets", "stopwords")


def invert_index(doc_word_counters):
    """Get word as key, all docs containing word as value"""
    index = defaultdict(list)
    for doc, counter in doc_word_counters.items():
        for word in counter.keys():
            index[word].append(doc)
    return index


def kl_div(p_x, p_c):
    return p_x * log(p_x / p_c, 2)


class KLDivergence:
    def __init__(self, word_counter, doc_word_counters):
        """
        F: Term (word) frequency
        L_x: Frequency sum of all docs containing word
        P_c: Distribution of word in the collection
        P_x: Distribution of word in sampled docs
        """
        self.words = word_counter.keys()
        self.F = [word_counter.get(word) for word in self.words]
        self.index = invert_index(doc_word_counters)
        self.L_x = [sum(doc_word_counters[doc].values()) for word in self.words for doc in self.index[word]]
        self.P_c = [float(F) / sum(word_counter.values()) for F in self.F]
        self.P_x = [float(self.F[i]) / self.L_x[i] for i in range(len(self.F))]
        self.values = [kl_div(self.P_x[i], self.P_c[i]) for i in range(len(self.P_x))]

    def write(self):
        words = [x for _, x in sorted(zip(self.values, self.words), reverse=True)][:70]  # Top 70
        content = "\n".join([s for s in words])
        target_file = join(STOPWORDS_FOLDER, "stopwords_dev.txt")
        with open(target_file, "w") as f:
            f.write(content)


def main():
    raw_file = join(COL_FOLDER, "corpus", "raw", "202108.txt")
    dataset = UDDataset.load_from_raw_file(raw_file)

    analyzer = UDAnalyzer()
    word_counter = analyzer.analyze_words(dataset)
    doc_word_counters = analyzer.get_doc_word_counters(dataset)

    KLDivergence(word_counter, doc_word_counters).write()


if __name__ == "__main__":
    main()
