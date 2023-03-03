from os.path import join


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
            is_valid = True
            tagged_sentence = []
            for row in s.split("\n"):
                tokens = row.split("\t")
                tokens = [token.strip() for token in tokens]
                tagged_sentence.append(tokens)
            for row in tagged_sentence:
                if (len(row[0])) == 0:
                    is_valid = False
            if is_valid:
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


def preprocess_vlsp2013(dataset):
    output = []
    for s in dataset:
        si = []
        for row in s:
            token, tag = row
            tag = "B-" + tag
            si.append([token, tag])
        output.append(si)
    return output
