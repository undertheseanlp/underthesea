from underthesea.corpus import Corpus
from underthesea.corpus.document import Document
from os.path import join
from os import listdir, mkdir

from underthesea.feature_engineering.unicode import UnicodeTransformer
from underthesea.util.file_io import write
import io


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
        contents = []
        for file in files:
            with io.open(file, "r", encoding="utf-8") as f:
                contents.append(f.read())
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
        except Exception:
            pass
        for document in self.documents:
            filename = join(folder, document.id)
            content = u"\n".join(document.sentences)
            write(filename, content)
