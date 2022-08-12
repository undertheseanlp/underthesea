from os.path import dirname, join

from underthesea import text_normalize


def add_lemma_column(sentence):
    rows = sentence.split("\n")
    result = []
    for row in rows:
        if row.startswith("# "):
            result.append(row)
        else:
            tokens = row.split("\t")
            lemma = text_normalize(tokens[0], tokenizer='space')
            if lemma != tokens[0]:
                print(f"{tokens[0]} -> {lemma}")
            tokens.insert(1, lemma)
            new_row = "\t".join(tokens)
            result.append(new_row)
    new_sentence = "\n".join(result)
    return new_sentence


filepath = join(dirname(dirname(__file__)), "corpus", "ud", "202108.txt")
text = open(filepath).read()
sentences = text.split("\n\n")
sentences = [add_lemma_column(s) for s in sentences]
new_text = "\n\n".join(sentences)
with open(join("tmp", "new_ud.txt"), "w") as f:
    f.write(new_text)
