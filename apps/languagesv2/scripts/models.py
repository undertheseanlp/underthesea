class Word:
    def __init__(self, id, text, freq):
        self.id = id
        self.text = text
        self.freq = freq

    def __repr__(self):
        return f"Word(id={self.id}, text='{self.text}', freq={self.freq})"
