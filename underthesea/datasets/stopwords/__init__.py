from os.path import join, dirname

filepath = join(dirname(__file__), "stopwords.txt")
with open(filepath, "r") as f:
    words = f.read().split("\n")
    f.close()
