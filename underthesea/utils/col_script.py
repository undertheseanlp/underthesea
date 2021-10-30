from os.path import dirname, join, exists
from torch.utils.data import Dataset
from underthesea import pos_tag
from underthesea import dependency_parse
from underthesea.utils.col_analyzer import UDAnalyzer

BOT_VERSION = "underthesea.v1.3.2"
PROJECT_FOLDER = dirname(dirname(dirname(__file__)))
DATASETS_FOLDER = join(PROJECT_FOLDER, "datasets")
COL_FOLDER = join(DATASETS_FOLDER, "UD_Vietnamese-COL")


class UDSentence:
    def __init__(self, rows, headers=None):
        self.rows = rows
        self.key_orders = ["doc_url", "date", "sent_id", "type", "authors", "text"]
        self.headers = headers

    def __str__(self):
        content = ""
        for key in self.key_orders:
            value = self.headers[key]
            content += f"# {key} = {value}\n"
        content += self.get_ud_str()
        return content

    def get_ud_str(self):
        return "\n".join(["\t".join(row) for row in self.rows])

    @staticmethod
    def _extract_header(content):
        content = content[2:].strip()
        index = content.find("=")
        key = content[:index].strip()
        value = content[index + 2:].strip()
        return [key, value]

    @staticmethod
    def load(content):
        data = content.split("\n")
        headers = [row for row in data if row.startswith("# ")]
        headers = dict([UDSentence._extract_header(content) for content in headers])
        rows = [row for row in data if not row.startswith("# ")]
        rows = [r.split("\t") for r in rows]
        s = UDSentence(rows=rows, headers=headers)
        return s

    @staticmethod
    def load_from_raw_content(raw_content):
        sentences = raw_content.split("\n")
        headers = sentences[:3]
        headers = dict([UDSentence._extract_header(_) for _ in headers])
        text = sentences[-1]
        headers["text"] = text
        headers["type"] = "bronze"
        headers["authors"] = BOT_VERSION
        pos_tags = pos_tag(text)
        dp_tags = dependency_parse(text)
        rows = [(item[0][0], item[0][1], str(item[1][1]), item[1][2]) for item in zip(pos_tags, dp_tags)]
        s = UDSentence(rows=rows, headers=headers)
        return s

    @staticmethod
    def load_from_raw_text(text):
        raw_content = ''
        raw_content += "# doc_url = DOC_URL\n"
        raw_content += "# date = DATE\n"
        raw_content += "# sent_id = 0\n"
        raw_content += text.strip()
        return UDSentence.load_from_raw_content(raw_content)


class UDDataset(Dataset):

    def __init__(self, sentences):
        super().__init__()
        self.sentences = sentences
        self.generate_indices()

    def generate_indices(self):
        self.sent_id_sent_index_map = dict([[s.headers["sent_id"], key] for key, s in enumerate(self.sentences)])

    def get_by_sent_id(self, sent_id):
        if sent_id in self.sent_id_sent_index_map:
            index = self.sent_id_sent_index_map[sent_id]
            return self.sentences[index]
        return None

    def __getitem__(self, index):
        return self.sentences[index]

    def __len__(self):
        return len(self.sentences)

    @staticmethod
    def load(ud_file):
        sentences = open(ud_file).read().split("\n\n")
        sentences = [UDSentence.load(s) for s in sentences]
        dataset = UDDataset(sentences)
        return dataset

    @staticmethod
    def load_from_raw_file(raw_file):
        with open(raw_file) as f:
            rows = f.read().strip().split("\n\n")
        sentences = [UDSentence.load_from_raw_content(content) for content in rows]
        dataset = UDDataset(sentences)
        return dataset

    def merge_sentence(self, s1, dataset):
        sent_id = s1.headers['sent_id']
        target_sentence = dataset.get_by_sent_id(sent_id)
        if target_sentence is None:
            return s1
        if target_sentence.headers['type'] == 'silver':
            return target_sentence
        return s1

    def merge(self, dataset):
        self.sentences = [self.merge_sentence(s, dataset) for s in self.sentences]

    def write(self, target_file):
        content = "\n\n".join([str(s) for s in self.sentences])
        with open(target_file, "w") as f:
            f.write(content)


if __name__ == '__main__':
    file = "202108.txt"  # lyrics.txt or 202108.txt
    raw_file = join(COL_FOLDER, "corpus", "raw", file)
    generated_dataset = UDDataset.load_from_raw_file(raw_file)

    current_file = join(COL_FOLDER, "corpus", "ud", file)
    if exists(current_file):
        current_dataset = UDDataset.load(current_file)

        generated_dataset.merge(current_dataset)

    target_file = join(COL_FOLDER, "corpus", "ud", file)
    generated_dataset.write(target_file)

    analyzer = UDAnalyzer()
    analyzer.analyze(generated_dataset)
    analyzer.analyze_today_words(generated_dataset)
