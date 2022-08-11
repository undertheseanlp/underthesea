from os.path import join
from tools import vtt
from tools import nvh

from underthesea.file_utils import DATASETS_FOLDER


def vtt_predict():
    def vtt_predict_sentence(s):
        rows = s.split("\n")
        result = []
        for row in rows:
            if row.startswith("# "):
                result.append(row)
            else:
                tokens = row.split("\t")
                tokens[1] = vtt.normalize(tokens[0])
                if tokens[1] != tokens[0]:
                    print(f"{tokens[0]} -> {tokens[1]}")
                new_row = "\t".join(tokens)
                result.append(new_row)
        new_s = "\n".join(result)
        return new_s

    corpus = join(DATASETS_FOLDER, "UD_Vietnamese-UUD-1.0.1-alpha", "all.txt")
    predict_file = join(DATASETS_FOLDER, "UD_Vietnamese-UUD-1.0.1-alpha", "predict.txt")
    with open(corpus) as f:
        text = f.read()
    sentences = text.split("\n\n")
    sentences = [vtt_predict_sentence(s) for s in sentences]
    new_text = "\n\n".join(sentences)
    with open(predict_file, "w") as f:
        f.write(new_text)
    print("Predict done")


def nvh_predict():
    def nvh_predict_sentence(s):
        rows = s.split("\n")
        result = []
        for row in rows:
            if row.startswith("# "):
                result.append(row)
            else:
                tokens = row.split("\t")
                tokens[1] = nvh.normalize(tokens[0])
                if tokens[1] != tokens[0]:
                    print(f"{tokens[0]} -> {tokens[1]}")
                new_row = "\t".join(tokens)
                result.append(new_row)
        new_s = "\n".join(result)
        return new_s

    corpus = join(DATASETS_FOLDER, "UD_Vietnamese-UUD-1.0.1-alpha", "all.txt")
    predict_file = join(DATASETS_FOLDER, "UD_Vietnamese-UUD-1.0.1-alpha", "predict.txt")
    with open(corpus) as f:
        text = f.read()
    sentences = text.split("\n\n")
    sentences = [nvh_predict_sentence(s) for s in sentences]
    new_text = "\n\n".join(sentences)
    with open(predict_file, "w") as f:
        f.write(new_text)
    print("Predict done")


def evaluate():
    corpus = join(DATASETS_FOLDER, "UD_Vietnamese-UUD-1.0.1-alpha", "all.txt")
    predict_file = join(DATASETS_FOLDER, "UD_Vietnamese-UUD-1.0.1-alpha", "predict.txt")
    with open(corpus) as f:
        text_true = f.read()
    with open(predict_file) as f:
        text_predict = f.read()
    sentences_true = text_true.split("\n\n")
    sentences_predict = text_predict.split("\n\n")
    total = 0.0
    correct = 0.0
    for i in range(len(sentences_true)):
        rows_true = sentences_true[i].split("\n")
        rows_predict = sentences_predict[i].split("\n")
        for j in range(len(rows_true)):
            if rows_true[j].startswith("#"):
                continue
            lemma_true = rows_true[j].split("\t")[1]
            lemma_predict = rows_predict[j].split("\t")[1]
            total += 1.0
            if lemma_true == lemma_predict:
                correct += 1.0
    accuracy = correct / total
    print("Evaluation done")
    print(f"Accuracy: {accuracy}")


if __name__ == '__main__':
    # vtt_predict()
    nvh_predict()
    evaluate()
