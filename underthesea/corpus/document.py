class Document:
    def __init__(self, id):
        """
        :param id id of document
        :type id: str
        """
        self.id = id
        self.content = None
        self.sentences = None

    def set_content(self, content):
        self.content = content

    def set_sentences(self, sentences):
        self.sentences = sentences
