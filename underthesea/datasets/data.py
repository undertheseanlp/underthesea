from os.path import join


class WordTokenizeCorpusReader:
    @staticmethod
    def read(data_folder, train_file=None, test_file=None):
        train = WordTokenizeCorpusReader.__read_data(join(data_folder, train_file))
        test = WordTokenizeCorpusReader.__read_data(join(data_folder, test_file))
        tagged_corpus = TaggedCorpus(train, test)
        return tagged_corpus

    @staticmethod
    def __read_data(data_file):
        text = open(data_file).read()
        sentences = text.split("\n")
        sentences = [WordTokenizeCorpusReader.__extract_tokens(s) for s in sentences]
        return sentences

    @staticmethod
    def __extract_tokens(s):
        sentence = []
        for item in s.split():
            tokens = item.split("_")
            tokens = [token for token in tokens if token]
            for i, token in enumerate(tokens):
                if i == 0:
                    sentence.append((token, "B-W"))
                else:
                    sentence.append((token, "I-W"))
        return sentence


class DataReader:
    @staticmethod
    def load_tagged_corpus(data_folder, train_file=None, test_file=None):
        train = DataReader.__read_tagged_data(join(data_folder, train_file))
        test = DataReader.__read_tagged_data(join(data_folder, test_file))
        tagged_corpus = TaggedCorpus(train, test)
        return tagged_corpus

    @staticmethod
    def __read_tagged_data(data_file):
        sentences = []
        raw_sentences = open(data_file).read().strip().split("\n\n")
        for s in raw_sentences:
            tagged_sentence = [node.split("\t") for node in s.split("\n") if not node.startswith("#")]
            sentences.append(tagged_sentence)
        return sentences


class TaggedCorpus:
    def __init__(self, train, test):
        self.train = train
        self.test = test

    def downsample(self, percentage):
        n = int(len(self.train) * percentage)
        self.train = self.train[:n]
        n = int(len(self.test) * percentage)
        self.test = self.test[:n]
        return self
