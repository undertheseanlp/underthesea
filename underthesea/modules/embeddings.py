import torch.nn


class WordEmbeddings(torch.nn.Module):
    def __init__(
        self,
        n_word: int = 100,
        learn=False
    ):
        if learn:
            return
        self.n_word = n_word

    @classmethod
    def fit(cls, sentences):
        n_word = 100
        embed = cls(
            n_word=n_word
        )
        return embed


class CharacterEmbeddings(torch.nn.Module):
    def __init__(self):
        pass

    def fit(self, sentences):
        pass


class FieldEmbeddings:
    def __init__(self):
        self._n_vocab = None

    @property
    def n_vocab(self) -> int:
        return self._n_vocab

    @n_vocab.setter
    def n_vocab(self, val):
        self._n_vocab = val

    def __repr__(self):
        s = f"{self.__class__.__name__}(n_vocab={self._n_vocab})"
        return s
