from os.path import dirname, join

filepath = join(dirname(dirname(__file__)), "corpus", "ud", "UUD_1.0.1-alpha.txt")
text = open(filepath).read()
sentences = text.split("\n\n")
print("Sentences:", len(sentences))
