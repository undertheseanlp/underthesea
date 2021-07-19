# CONTENT
# #1
# Mà không biết phải nhân viên cũ hay mới nữa nhưng cảm giác thân thiện hơn.
# {SERVICE#GENERAL, positive}
#
# #2
# Nay đi uống mới biết giá thành hơi cao nhưng thật sự đi đôi với chất lượng.
# {RESTAURANT#PRICES, negative}, {RESTAURANT#GENERAL, positive}
#

fnames = ["/Users/taidnguyen/Desktop/Sentence-level-Restaurant/Dev.txt",
          "/Users/taidnguyen/Desktop/Sentence-level-Hotel/Dev_Hotel.txt"]


class ABSentimentData(str):
    def __init__(self, fname):
        fin = open(fname, mode="r", encoding="utf-8", errors="ignore")
        lines = fin.readlines()
        fin.close()

        data = []
        labels = []
        for i in range(1, len(lines), 4):
            sentence = lines[i].replace("\n", "")
            label = lines[i + 1].replace("\n", "")
            data.append(sentence)
            labels.append(label)

        self.data = data
        self.labels = labels  #aspect, sub-aspect, tone

        # print(self.data)
        # print(self.labels)


class Tokenizer(object):
    def __init__(self, max_seq_len, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len




ABSentimentData(fnames[1])
