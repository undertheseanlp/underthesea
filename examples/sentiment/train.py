# Txt format:
# #1
# Mà không biết phải nhân viên cũ hay mới nữa nhưng cảm giác thân thiện hơn.
# {SERVICE#GENERAL, positive}
#
# #2
# Nay đi uống mới biết giá thành hơi cao nhưng thật sự đi đôi với chất lượng.
# {RESTAURANT#PRICES, negative}, {RESTAURANT#GENERAL, positive}
#

import re

fnames = ["/Users/taidnguyen/Desktop/Sentence-level-Restaurant/Dev.txt",
          "/Users/taidnguyen/Desktop/Sentence-level-Hotel/Dev_Hotel.txt"]


class ABSentimentData(str):
    def __init__(self, fname):
        fin = open(fname, mode="r", encoding="utf-8", errors="ignore")
        lines = fin.readlines()
        fin.close()

        sentences = []
        aspects = []
        subaspects = []
        tones = []
        for i in range(1, len(lines), 4):
            sentence = lines[i].replace("\n", "")
            label_pattern = r"([\w#&]+)"  # keep aspect like DESIGN&FEATURES together
            labels = re.findall(label_pattern,
                                lines[i + 1].replace("\n", ""))  # array of 'ASPECT#SUBASPECT', 'tone',... for sentence

            aspect = []
            subaspect = []
            tone = []
            for j in range(0, len(labels), 2):
                aspect.append(labels[j].split("#")[0])
                subaspect.append(labels[j].split("#")[1])
                tone.append(labels[j + 1])
            aspects.append(aspect)
            subaspects.append(subaspect)
            tones.append(tone)
            sentences.append(sentence)

        assert len(sentences) == len(aspects) == len(subaspects) == len(tones), \
            "Mismatched data while splitting:\n{0}, {1}, {2}, {3}".format(len(sentences), len(aspects), len(subaspects),
                                                                          len(tones))

        self.sentences = sentences
        self.aspects = aspects
        self.subaspects = subaspects
        self.tones = tones

        print(self.sentences[1], self.aspects[1], self.subaspects[1], self.tones[1])


class Tokenizer(object):
    def __init__(self, max_seq_len, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len


ABSentimentData(fnames[0])
