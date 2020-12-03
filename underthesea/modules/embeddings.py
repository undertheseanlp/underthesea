class WordEmbedding:
    def __init__(
        self,
        n_word: int = 10,
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


class CharacterEmbedding:
    def __init__(self):
        pass

    def fit(self, sentences):
        pass
