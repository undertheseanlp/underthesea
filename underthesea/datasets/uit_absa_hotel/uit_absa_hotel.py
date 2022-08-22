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

    def __init__(self, training="aspect"):
        DataFetcher.download_data(UITABSAHotel.NAME, None)
        train_file = join(DATASETS_FOLDER, UITABSAHotel.NAME, "Train_Hotel.txt")
        dev_file = join(DATASETS_FOLDER, UITABSAHotel.NAME, "Dev_Hotel.txt")
        test_file = join(DATASETS_FOLDER, UITABSAHotel.NAME, "Test_Hotel.txt")
        print("Currently training: %s (aspect or polarity)" % training)

        self.training = training  # aspect or polarity
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
        aspect_labels = [label[0] for label in labels]
        polarity_labels = [label[1] for label in labels]  # some odd polarity labels outside of pos, neg, neutral

        if self.training == "aspect":
            label_ids = self.label_encoder.encode(aspect_labels)
        else:
            label_ids = self.label_encoder.encode(polarity_labels)

        return {
            "sentence_id": sentence_id,
            "text": text,
            "labels": labels,
            "aspect_labels": aspect_labels,
            "polarity_labels": polarity_labels,
            "label_ids": label_ids
        }

    def _extract_sentences(self, file):
        with open(file, encoding='utf-8') as f:
            content = f.read()
            sentences = content.split("\n\n")
            sentences = [self._extract_sentence(s) for s in sentences]
        return sentences


if __name__ == '__main__':
    dataset = UITABSAHotel(training="aspect")
