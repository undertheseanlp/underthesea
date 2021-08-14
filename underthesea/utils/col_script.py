from os.path import dirname, join
from torch.utils.data import Dataset
from underthesea import pos_tag
from underthesea import dependency_parse
from underthesea.utils.col_analyzer import UDAnalyzer

BOT_VERSION = "underthesea.v1.3.2"
PROJECT_FOLDER = dirname(dirname(dirname(__file__)))
DATASETS_FOLDER = join(PROJECT_FOLDER, "datasets")
COL_FOLDER = join(DATASETS_FOLDER, "UD_Vietnamese-COL")


class Sentence(object):
    def __init__(self, content):
        sentences = content.split("\n")
        self.headers = sentences[:3]
        self.headers.append("# type = bronze")
        self.headers.append(f"# authors = {BOT_VERSION}")
        self.url = sentences[0][len("# doc_url = "):]
        self.date = sentences[1][len("# date = "):]
        self.sent_id = sentences[2][len("# sent_id = "):]
        self.text = sentences[-1]
        self.headers.append("# text = " + self.text)
        self.ud_content = ""

    def get_tags(self):
        pos_tags = pos_tag(self.text)
        dp_tags = dependency_parse(self.text)
        self.tags = [(item[0][0], item[0][1], str(item[1][1]), item[1][2]) for item in zip(pos_tags, dp_tags)]
        return self.tags

    def to_ud(self):
        tags = self.get_tags()
        ud_content = "\n".join(["\t".join(tag) for tag in tags])
        self.ud_content = "\n".join(self.headers) + "\n"
        self.ud_content += ud_content
        return self


class RawToUDDataset(Dataset):
    def __init__(self, raw_file):
        super().__init__()
        self.len = 1
        with open(raw_file) as f:
            sentences = f.read().strip().split("\n\n")
        sentences = [Sentence(s) for s in sentences]
        self.sentences = [s.to_ud() for s in sentences]
        self.len = len(self.sentences)

    def __getitem__(self, index):
        return self.sentences[index]

    def __len__(self):
        return self.len

    def write(self, filepath):
        with open(filepath, "w") as f:
            content = "\n\n".join([s.ud_content for s in self])
            f.write(content)


if __name__ == '__main__':
    raw_file = join(COL_FOLDER, "corpus", "raw", "202108.txt")
    dataset = RawToUDDataset(raw_file)

    ud_file = join(COL_FOLDER, "corpus", "ud", "202108.txt")
    dataset.write(ud_file)

    analyzer = UDAnalyzer()
    analyzer.analyze(dataset)
