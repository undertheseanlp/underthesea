from functools import reduce

from languageflow.transformer import Transformer


class WordVectorTransformer(Transformer):
    """ Convert a collection of raw documents to a matrix of text-index
    features

    keep the order of word in document
    index start with 1, index 0 is default padding_value

    Parameters
    ----------

    embedding_size: size of embedding word to vector
    """

    def __init__(self, embedding_size=100, padding=False, padding_value=0,
                 unknown_value=1):
        self.vocab = []
        self.embedding_size = embedding_size
        self.padding = padding
        self.padding_value = padding_value
        self.unknown_value = unknown_value

        self.max_length = 0
        self.ns = 2  # number of special values

    def get_vocab_size(self):
        # Add 2 tokens for
        # - padding_value (index 0)
        # - unknown_value (index 1)
        return len(self.vocab) + self.ns

    def _doc2index(self, document):
        output = [self.word2index[token]
                    if token in self.vocab else self.unknown_value for
                  token in document]
        if self.padding == 'max':
            if len(output) < self.max_length:
                output += [self.padding_value] * (self.max_length - len(output))
            else:
                output = output[:self.max_length]
        return output

    def _update_vocab(self, vocab):
        new_words = sorted(set(vocab) - set(self.vocab))
        self.vocab = self.vocab + new_words
        n = len(self.vocab)
        self.word2index = dict(zip(self.vocab, range(self.ns, n + self.ns)))
        self.index2word = dict(zip(range(self.ns, n + self.ns), self.vocab))

    def _transform(self, raw_documents, is_fit):
        documents = [document.split() for document in raw_documents]
        vocab = sorted(set((reduce(lambda x, y: x + y, documents))))
        max_length_doc = max(len(_) for _ in documents)
        if is_fit:
            self.max_length = max(self.max_length, max_length_doc)
            self._update_vocab(vocab)
        output = [self._doc2index(document) for document in documents]
        return output

    def fit_transform(self, raw_documents):
        return self._transform(raw_documents, is_fit=True)

    def transform(self, raw_documents):
        return self._transform(raw_documents, is_fit=False)
