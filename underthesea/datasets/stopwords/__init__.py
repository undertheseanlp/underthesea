from os.path import dirname, join

filepath = join(dirname(__file__), "stopwords.txt")
with open(filepath) as f:
    words = f.read().split("\n")
    f.close()
