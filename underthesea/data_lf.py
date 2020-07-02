from pathlib import Path
from typing import List, Union


class Corpus:
    pass


class Label:
    def __init__(self, value: str, score: float = 1.0):
        self.value = value
        self.score = score

    @property
    def value(self):
        return self._value

    @property
    def score(self):
        return self._score

    @value.setter
    def value(self, value):
        self._value = value

    @score.setter
    def score(self, score):
        if 0.0 <= score <= 1.0:
            self._score = score
        else:
            self._score = 1.0

    def __str__(self):
        return "{} ({})".format(self._value, self._score)

    def __repr__(self):
        return "{} ({})".format(self._value, self._score)


class Sentence:
    def __init__(
        self,
        text: str = None,
        labels: Union[List[Label], List[str]] = None
    ):
        self.text = text
        self.labels = labels

    def __str__(self) -> str:
        return f'Sentence: "{self.text}" - Labels: {self.labels}'

    def __repr__(self) -> str:
        return f'Sentence: "{self.text}" - Labels: {self.labels}'

    def to_text_classification_format(self) -> str:
        labels_text = " ".join([f"__label__{label.value}" for label in self.labels])
        output = f"{labels_text} {self.text}"
        return output

    def add_labels(self, labels: Union[List[Label], List[str]]):
        for label in labels:
            if type(label) == str:
                label = Label(label)
            if not self.labels:
                self.labels = []
            self.labels.append(label)


class PlaintextCorpus(Corpus):
    def __init__(self, sentences):
        self.sentences = sentences


class CategorizedCorpus(Corpus):
    def __init__(
        self,
        train: List[Sentence],
        dev: List[Sentence],
        test: List[Sentence]
    ):
        self._train: List[Sentence] = train
        self._dev: List[Sentence] = dev
        self._test: List[Sentence] = test

    @property
    def train(self) -> List[Sentence]:
        return self._train

    @property
    def dev(self) -> List[Sentence]:
        return self._dev

    @property
    def test(self) -> List[Sentence]:
        return self._test

    def __str__(self) -> str:
        return "CategorizedCorpus: %d train + %d dev + %d test sentences" % (
            len(self.train),
            len(self.dev),
            len(self.test),
        )

    def save(self, data_folder: str):
        self.save_sentences(self._train, f"{data_folder}/train.txt")
        self.save_sentences(self._dev, f"{data_folder}/dev.txt")
        self.save_sentences(self._test, f"{data_folder}/test.txt")

    def save_sentences(self, sentences: List[Sentence], path_to_file: Union[Path, str]):
        with open(path_to_file, "w") as f:
            for sentence in sentences:
                f.write(sentence.to_text_classification_format() + "\n")


class TaggedCorpus(Corpus):
    pass
