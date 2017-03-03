from corpus import Corpus
from document import Document
from os.path import join
from os import listdir, mkdir



class PlainTextCorpus(Corpus):
    """class for handling plain text corpus
    """

    def __init__(self):
        self.documents = None

    def load(self, folder):
        """load plaintext folder to documents

        :param str folder: path to directory
        :type folder: string
        """
        ids = listdir(folder)
        files = [join(folder, f) for f in ids]
        contents = [open(f, "r").read() for f in files]
        documents = []

        for id, content in zip(ids, contents):
            document = Document(id)
            document.set_sentences(content)
            documents.append(document)
        self.documents = documents

    def save(self, folder):
        """save corpus to files

        :param str folder: path to directory
        :type folder: string
        """
        try:
            mkdir(folder)
        except Exception, e:
            pass
        for document in self.documents:
            f = join(folder, document.id)
            open(f, "w").write(document.sentences)
