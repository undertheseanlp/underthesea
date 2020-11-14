from pathlib import Path
from typing import List, Union
from underthesea.corpus import Corpus
from underthesea.corpus.data import Sentence


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
