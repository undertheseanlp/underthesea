from os.path import join
import re
from underthesea.data_fetcher import DataFetcher
from underthesea.file_utils import DATASETS_FOLDER

_CITATION = None

_DESCRIPTION = None

_HOMEPAGE = None

_LICENSE = None


class LabelEncoder:
    def __init__(self):
        self.index2label = {}
        self.label2index = {}
        self.vocab_size = 0

    def encode(self, labels):
        if type(labels) == list:
            return [self.encode(label) for label in labels]
        label = labels  # label is a string
        if label not in self.label2index:
            index = self.vocab_size
            self.label2index[label] = index
            self.index2label[index] = label
            self.vocab_size += 1
            return index
        else:
            return self.label2index[label]


class UITABSAHotel:
    NAME = "UIT_ABSA_HOTEL"
    VERSION = "1.0"

    def __init__(self):
        DataFetcher.download_data(UITABSAHotel.NAME, None)
        train_file = join(DATASETS_FOLDER, UITABSAHotel.NAME, "Train_Hotel.txt")
        dev_file = join(DATASETS_FOLDER, UITABSAHotel.NAME, "Dev_Hotel.txt")
        test_file = join(DATASETS_FOLDER, UITABSAHotel.NAME, "Test_Hotel.txt")

        self.label_encoder = LabelEncoder()

        self.train = self._extract_sentences(train_file)
        self.dev = self._extract_sentences(dev_file)
        self.test = self._extract_sentences(test_file)
        self.num_labels = self.label_encoder.vocab_size

    def _join_labels(self, label):
        return "#".join(label)

    def _extract_sentence(self, sentence):
        sentence_id, text, labels = sentence.split("\n")
        labels = re.findall("\{(?P<aspect>.*?), (?P<polarity>.*?)\}", labels)
        labels = [self._join_labels(label) for label in labels]
        label_ids = self.label_encoder.encode(labels)
        return sentence_id, text, labels, label_ids

    def _extract_sentences(self, file):
        with open(file) as f:
            content = f.read()
            sentences = content.split("\n\n")
            sentences = [self._extract_sentence(s) for s in sentences]
        return sentences


if __name__ == '__main__':
    dataset = UITABSAHotel()
