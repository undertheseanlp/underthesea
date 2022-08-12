from os.path import join
from tools import vtt
from tools import nvh
from tools import vtm
import hydra
from omegaconf import DictConfig, OmegaConf

from underthesea import text_normalize
from underthesea.file_utils import DATASETS_FOLDER


def predict_sentence(s, conf):
    model_name = conf["model"]
    rows = s.split("\n")
    result = []
    for row in rows:
        if row.startswith("# "):
            result.append(row)
        else:
            tokens = row.split("\t")
            if model_name == "vtt":
                tokens[1] = vtt.normalize(tokens[0])
            elif model_name == "nvh":
                tokens[1] = nvh.normalize(tokens[0])
            elif model_name == "uts":
                tokens[1] = text_normalize(tokens[0], tokenizer='space')
            elif model_name == "vtm":
                tokens[1] = vtm.normalize(tokens[0])
            if tokens[1] != tokens[0]:
                print(f"{tokens[0]} -> {tokens[1]}")
            new_row = "\t".join(tokens)
            result.append(new_row)
    new_s = "\n".join(result)
    return new_s


def predict(cfg):
    corpus = join(DATASETS_FOLDER, "UD_Vietnamese-UUD-1.0.1-alpha", "all.txt")
    predict_file = join(DATASETS_FOLDER, "UD_Vietnamese-UUD-1.0.1-alpha", "predict.txt")
    with open(corpus) as f:
        text = f.read()
    sentences = text.split("\n\n")

    sentences = [predict_sentence(s, cfg) for s in sentences]
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


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    predict(cfg)
    evaluate()


if __name__ == '__main__':
    # vtt_predict()
    my_app()
