from pathlib import Path
from typing import Union

from underthesea.corpus import Corpus
from underthesea.corpus.data import Sentence


class CategorizedCorpus(Corpus):
    def __init__(
        self,
        train: list[Sentence],
        dev: list[Sentence],
        test: list[Sentence]
    ):
        self._train: list[Sentence] = train
        self._dev: list[Sentence] = dev
        self._test: list[Sentence] = test

    @property
    def train(self) -> list[Sentence]:
        return self._train

    @property
    def dev(self) -> list[Sentence]:
        return self._dev

    @property
    def test(self) -> list[Sentence]:
        return self._test

    def __str__(self) -> str:
        return f"CategorizedCorpus: {len(self.train)} train + {len(self.dev)} dev + {len(self.test)} test sentences"

    def save(self, data_folder: str):
        self.save_sentences(self._train, f"{data_folder}/train.txt")
        self.save_sentences(self._dev, f"{data_folder}/dev.txt")
        self.save_sentences(self._test, f"{data_folder}/test.txt")

    def save_sentences(self, sentences: list[Sentence], path_to_file: Union[Path, str]):
        with open(path_to_file, "w") as f:
            for sentence in sentences:
                f.write(sentence.to_text_classification_format() + "\n")
