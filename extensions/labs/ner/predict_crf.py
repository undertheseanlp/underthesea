from os.path import join

import hydra
import joblib
from hydra.utils import get_original_cwd
from underthesea_core import CRFTagger

from underthesea import pos_tag


@hydra.main(version_base=None)
def my_app(cfg):
    working_dir = get_original_cwd()
    print("working_dir", working_dir)

    if "output_dir" not in cfg:
        output_dir_path = "tmp/ner"
    else:
        output_dir_path = cfg["output_dir"]
    output_dir = join(working_dir, output_dir_path)

    if "text" not in cfg:
        text = "Quỳnh Như tiết lộ với báo Bồ Đào Nha về hành trình làm nên lịch sử"
    else:
        text = cfg["text"]

    tokens = pos_tag(text)

    model = CRFTagger()
    model.load(join(output_dir, "models.bin"))
    features = joblib.load(join(output_dir, "features.bin"))
    dictionary = joblib.load(join(output_dir, "dictionary.bin"))
    model.set_featurizer(features, dictionary)
    y = model.predict(tokens)
    for token, x in zip(tokens, y):
        print(token, "\t", x)


if __name__ == '__main__':
    my_app()
