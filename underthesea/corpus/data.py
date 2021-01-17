from typing import List, Union


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


class Document:
    def __init__(self, id):
        """
        :param id id of document
        :type id: str
        """
        self.id = id
        self.content = None
        self.sentences = None

    def set_content(self, content):
        self.content = content

    def set_sentences(self, sentences):
        self.sentences = sentences
