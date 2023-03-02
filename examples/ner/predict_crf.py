import hydra

from os.path import dirname, join
from underthesea.models.fast_crf_sequence_tagger import FastCRFSequenceTagger
from underthesea import word_tokenize
from underthesea import pos_tag

working_dir = dirname(__file__)

@hydra.main()
def my_app(cfg):
    # check if output_dir in dict cfg
    "output_dir" 

    if not "output_dir" in cfg:
        output_dir_path = "tmp/ner" 
    else:
        output_dir_path = cfg["output_dir"]
    output_dir = join(working_dir, output_dir_path)

    if not "text" in cfg:
        text = "Quỳnh Như tiết lộ với báo Bồ Đào Nha về hành trình làm nên lịch sử"
    else:
        text = cfg["text"]

    tokens = pos_tag(text)

    model = FastCRFSequenceTagger()
    model.load(output_dir)
    y = model.predict(tokens)
    for token, x in zip(tokens, y):
        print(token, "\t", x)

if __name__ == '__main__':
    my_app()
