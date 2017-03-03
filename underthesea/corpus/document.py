class Document:
    def __init__(self, id):
        """
        :param id id of document
        :type id: str
        """
        self.id = id
        self.sentences = None

    def set_sentences(self, sentences):
        self.sentences = sentences
