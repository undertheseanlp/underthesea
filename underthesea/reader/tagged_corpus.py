from os import mkdir
from os.path import join
import pandas as pd
import io

import shutil
from languageflow.util.file_io import read, write


class TaggedCorpus:
    def __init__(self, sentences=[]):
        self.sentences = sentences

    def save(self, filepath):
        sentences = self.sentences
        content = "\n\n".join(
            ["\n".join(["\t".join(t) for t in s]) for s in sentences])
        with io.open(filepath, "w", newline="\n", encoding="utf-8") as f:
            f.write(content)

    def _parse_sentence(self, text, skip_column=None):
        tokens = text.split("\n")
        if skip_column is None:
            tokens = [token.split("\t") for token in tokens if str(token).strip() != ""]
        else:
            tokens = [token.split("\t")[(skip_column + 1):] for token in tokens if str(token).strip() != ""]
        return tokens

    def load(self, filepath, skip_column=None):
        content = read(filepath).strip()
        sentences = content.split("\n\n")
        sentences = [self._parse_sentence(str(s).strip(), skip_column) for s in sentences if str(s).strip() != ""]
        self.sentences = sentences

    def _analyze_field(self, df, id, output_folder=".", n_head=10):
        id = str(id)
        m = df.shape[1]
        df.columns = [str(i) for i in range(m)]

        agg_dict = dict()
        agg_dict[id] = "size"
        for i in range(int(id)):
            agg_dict[str(i)] = lambda x: ", ".join(
                pd.value_counts(x).index[:n_head])
        name_dict = dict()
        name_dict[id] = "count"
        df_analyze = df.groupby(id).agg(agg_dict).rename(
            columns=name_dict).reset_index()
        filename = join(output_folder, "column-%s-analyze.xlsx" % id)

        log = u""
        log += u"Tags         : {}\n".format(df_analyze.shape[0])
        tags = df_analyze[id].to_dict().values()
        tags = sorted(tags)
        log += u"List tags    : {}\n".format(u", ".join(tags))
        df_analyze.to_excel(filename, index=False)
        return log

    def _analyze_first_token(self, df, id, output_folder="."):
        filename = join(output_folder, "column-%s-analyze.xlsx" % id)
        df_analyze = df[id].value_counts().reset_index(name="count")
        df_analyze = df_analyze.rename(columns={"index": "0"})
        df_analyze.to_excel(filename, index=False)
        log = u""
        log += u"Unique words : {}\n".format(df_analyze.shape[0])
        log += u"Top words    : {}\n".format(
            u", ".join(list(df_analyze["0"].to_dict().values())[:20]))
        return log

    def analyze(self, output_folder=".", auto_remove=False):
        """
        :type auto_remove: boolean
        :param boolean auto_remove: auto remove previous files in analyze folder
        """
        if auto_remove:
            try:
                shutil.rmtree(output_folder)
            except Exception:
                pass
        try:
            mkdir(output_folder)
        except Exception:
            pass
        tokens = [token for sublist in self.sentences for token in sublist]
        df = pd.DataFrame(tokens)
        log = u""
        log += u"Sentences    : {}\n".format(len(self.sentences))
        n = df.shape[1]
        log += self._analyze_first_token(df, 0, output_folder)
        for i in range(1, n):
            log += self._analyze_field(df, i, output_folder)
        print(log)
        stat_file = join(output_folder, "stats.txt")
        write(stat_file, log)
