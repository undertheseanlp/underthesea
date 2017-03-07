from corpus import Corpus
from underthesea.corpus.document import Document
from os.path import join
from os import listdir, mkdir

from underthesea.transformer.unicode import UnicodeTransformer


class PlainTextCorpus(Corpus):
    """class for handling plain text corpus
    """

    def __init__(self):
        self.documents = None

    def load(self, folder):
        """load plaintext folder to documents and apply unicode transformer

        :param str folder: path to directory
        :type folder: string
        """
        ids = listdir(folder)
        files = [join(folder, f) for f in ids]
        contents = [open(f, "r").read() for f in files]
        documents = []

        for id, content in zip(ids, contents):
            document = Document(id)
            sentences = content.split("\n")
            unicode_transformer = UnicodeTransformer()
            sentences = [unicode_transformer.transform(sentence) for sentence in sentences]
            document.set_sentences(sentences)
            documents.append(document)
        self.documents = documents

    def save(self, folder):
        """save corpus to files

        :param str folder: path to directory
        :type folder: string
        """
        try:
            mkdir(folder)
        except Exception as e:
            pass
        for document in self.documents:
            f = join(folder, document.id)
            content = u"\n".join(document.sentences)
            content = content.encode("utf-8")
            open(f, "w").write(content)
