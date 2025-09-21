from os.path import join

max_len = 0


def transform(src_file, dest_file):
    def transform_line(line):
        tokens = line.split("\t")
        output = tokens[0].replace(" ", "_") + " " + tokens[-1]
        return output

    def transform_sentence(sentence):
        global max_len
        lines = sentence.split("\n")
        max_len = max(max_len, len(lines))
        lines = [transform_line(line) for line in lines]
        sentence = "\n".join(lines)
        return sentence

    with open(src_file) as f:
        content = f.read()
    sentences = content.split("\n\n")
    sentences = [transform_sentence(s) for s in sentences]
    content = "\n\n".join(sentences)
    with open(dest_file, "w") as f:
        f.write(content)


for file in ["train.txt", "test.txt", "dev.txt"]:
    print(file)
    max_len = 0
    src_folder = join("bert-ner", "data", "vlsp2016")
    dest_folder = join("bert-ner", "data")
    transform(src_file=join(src_folder, file), dest_file=join(dest_folder, file))
    print(max_len)
