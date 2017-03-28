from underthesea.corpus import Document, UnicodeTransformer
from underthesea.underthesea import Corpus
from os import listdir, mkdir
from os.path import join


class WSCorpus(Corpus):
    def __init__(self):
        pass

    def map_token(self, token):
        if token[1] == "B_W":
            return " " + token[0]
        if token[1] == "I_W":
            return "_" + token[0]
        else:
            return " " + token[0]

    def _column_to_text(self, sentence):
        s = sentence
        tokens = s.split("\n")
        tokens = [t.split("\t") for t in tokens]
        tokens = [self.map_token(t) for t in tokens]
        s = "".join(tokens)
        s = s.strip()
        return s

    def load(self, folder, format):
        """

        :param str folder: path to directory
        :type folder: string
        :param str format: either TEXT or COLUMN
        :type format: str
        """
        ids = listdir(folder)
        files = [join(folder, file) for file in ids]
        contents = [open(f, "r").read().strip() for f in files]
        documents = []

        for id, content in zip(ids, contents):
            print id
            document = Document(id)
            unicode_transformer = UnicodeTransformer()
            content = unicode_transformer.transform(content)
            sentences = content.split("\n\n")
            sentences = [self._column_to_text(s) for s in sentences]
            document.set_sentences(sentences)
            documents.append(document)

        self.documents = documents

    def save(self, folder, format):
        """save wscorpus to files

        :param str folder: path to directory
        :type folder: string
        :param str format: either TEXT or COLUMN
        :type format: str
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
