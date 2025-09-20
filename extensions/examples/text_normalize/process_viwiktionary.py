from os.path import join
# from bs4 import BeautifulSoup

from underthesea.file_utils import DATASETS_FOLDER
from underthesea.pipeline.word_tokenize.regex_tokenize import VIETNAMESE_CHARACTERS_LOWER, VIETNAMESE_VOWELS_LOWER
import re

VIWIK_FOLDER = join(DATASETS_FOLDER, "viwiktionary")

valid_rime_patterns = [
    r"[aàáảãạeèéẻẽẹêềếểễệiìíỉĩịoòóỏõọôồốổỗộơờớởỡợuùúủũụưừứửữựyỳýỷỹỵ]",

    # "oa", "oe", "uê", "uy" có 2 cách bỏ dấu
    #  - bỏ dấu kiểu cũ
    r"[oòóỏõọ][ae]",
    r"[uùúủũụ][êy]",

    # - bỏ dấu kiểu mới
    r"o[aàáảãạeèéẻẽẹ]",
    r"u[êềếểễệyỳýỷỹỵ]",

    # normal: includes 2 groups:
    # - [cpt]|ch       : grave/dot (sắc/nặng)
    r"([ắặấậéẹóọúụ]|i[ếệ]|o?[áạ]|u?[ốộ]|ư[ớợ])[cpt]",
    r"[ếệíịớợ][pt]",
    r"([ứự]|o[ắặ])[ct]",
    r"o[óọ]c",
    r"(o[éẹ]|u[ấậ]|uy[ếệ]?)t",
    r"([ếệíị]|o?[áạ]|u[ếệýỵ])ch",

    # - [aimouy]|n[gh]?: every tone
    r"([iìíỉĩịuùúủũụưừứửữự]|uy)a",
    r"([oòóỏõọuùúủũụưừứửữự]|o?[aàáảãạ]|u?[ôồốổỗộ]|ư?[ơờớởỡợ])i",
    r"o?[aàáảãạeèéẻẽẹ]o",
    r"([aàáảãạâầấẩẫậiìíỉĩịưừứửữự]|u[yỳýỷỹỵ]|ư[ơờớởỡợ]|i?[êềếểễệ])u",
    r"(o?[aàáảãạ]|u?[âầấẩẫậ])y",
    r"([âầấẩẫậeèéẻẽẹoòóỏõọuùúủũụưừứửữự]|o?[aàáảãạăằắẳẵặ]|i[êềếểễệ]|u?[ôồốổỗộ]|ư[ơờớởỡợ])(m|ng?)",  # m, n, g
    r"([êềếểễệiìíỉĩịơờớởỡợ]|o[eèéẻẽẹ])[mn]",  # m, n
    r"u[âầấẩẫậ]ng?",  # n, ng
    r"uy[êềếểễệ]?n",  # n
    r"o[oòóỏõọ]ng",  # ng
    r"([êềếểễệiìíỉĩị]|o?[aàáảãạ]|u[êềếểễệyỳýỷỹỵ])nh"
]

with_tone = "àáảãạèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵằắẳẵặầấẩẫậ"
without_tone = "aaaaaeeeeeêêêêêiiiiioooooôôôôôơơơơơuuuuuưưưưưyyyyyăăăăăâââââ"

clearing_tone = str.maketrans(with_tone, without_tone)
yi_trans = str.maketrans("yỳýỷỹỵ", "iìíỉĩị")


def remove_tone(s):
    return s.translate(clearing_tone)


def get_rime(s):  # kiểm tra vần hợp lệ
    for pattern in valid_rime_patterns:
        if re.match(r"^" + pattern + r"$", s):
            return remove_tone(s)
    return False


def get_tone(s):
    if re.match(r".?[àằầèềìòồờùừỳ].*", s):
        return "acute"  # huyền

    if re.match(r".?[áắấéếíóốớúứý].*", s):
        return "grave"  # sắc

    if re.match(r".?[ảẳẩẻểỉỏổởủửỷ].*", s):
        return "hook"  # hỏi

    if re.match(r".?[ãẵẫẽễĩõỗỡũữỹ].*", s):
        return "tilde"  # ngã

    if re.match(r".?[ạặậẹệịọộợụựỵ].*", s):
        return "dot"  # nặng

    if re.match(r".?[aăâeêioôơuưy].*", s):
        return "blank"  # ngang

    return None


# flake8: noqa: C901
def split_vietnamese_syllable(s):
    onset, right_side, tone = None, None, None

    if re.match(r"^[aàáảãạăằắẳẵặâầấẩẫậeèéẻẽẹêềếểễệiìíỉĩịoòóỏõọôồốổỗộơờớởỡợuùúủũụưừứửữựyỳýỷỹỵ]", s):  # no leading onset
        onset = ""
        if re.match(r"^[yỳýỷỹỵ]", s):  # s = y_
            if re.match(r"^y[êềếểễệ]([mu]|ng?)|y[ếệ]t$", s):  # s = yê_
                right_side = "i" + s[1:]
        else:
            right_side = s

    elif s.startswith("qu"):  # `qu_` group
        onset = "qu"

        if re.match(r"^qu[ốộ]c$", s):  # u_ rime
            right_side = s[1:]

        else:  # tính vần đằng sau
            if re.match(r"^qu[yỳýỷỹỵ].*$", s):  # "quy.*"  # -> i* rime
                right_side = s[2].translate(yi_trans) + s[3:]
            else:  # "qu(.+)"
                right_side = s[2:]

    elif re.match(r"^g[iìíỉĩị]", s):  # `gi_` group
        onset = "gi"

        if re.match(r"^gi[êềếểễệ]([mu]|ng?)|gi[ếệ][cpt]|g[íị](p|ch)|g[iìíỉĩị]n?$", s):
            # i_ rime, gi_ = gi + i_
            # "giê([mu]|[cpt]|ng?)"  # -> iê_
            # "gin?", "gi(p|ch)"  # -> i*
            right_side = s[1:]

        else:  # rime comes after gi: "gi(.+)"
            right_side = s[2:]

    else:
        # all-rime group
        free_type = re.findall(r"^([bdđlmrsvx]|[cknp]?h|t[hr]?|n(?!g)|p(?!h))(.+)$", s)
        if free_type:
            onset, right_side = free_type[0]

        # eiy group ("eêiy"):
        eiy_type = re.findall(r"^(k|n?gh)([eèéẻẽẹêềếểễệiìíỉĩịyỳýỷỹỵ].*)$", s)
        if eiy_type:
            onset, right_side = eiy_type[0]

        # aou group ("aăâoôơuư"):
        aou_type = re.findall(r"^(c|n?g)([aàáảãạăằắẳẵặâầấẩẫậoòóỏõọôồốổỗộơờớởỡợuùúủũụưừứửữự].*)$", s)
        if aou_type:
            onset, right_side = aou_type[0]

    if onset is not None and right_side is not None:
        rime = get_rime(right_side)
        tone = get_tone(right_side)

        if rime:
            return (onset, rime, tone)


def read_wiki_dump():
    viwik_file = join(VIWIK_FOLDER, "viwiktionary-20220801-pages-articles-multistream.xml")
    with open(viwik_file, 'r') as f:
        data = f.read()
    # bs_data = BeautifulSoup(data, "xml")


class Word:
    def __init__(self, text):
        self.text = text.lower()
        self.syllables = self.text.split(" ")
        self.is_word = self.check_is_word()
        self.is_vietnamese = self.check_is_vietnamese()
        self.onset = None
        self.rime = None
        self.tone = None

    def check_is_word(self):
        if len(self.text) == 0:
            return False
        return True

    def check_is_vietnamese(self):
        if not self.is_word:
            return False
        for syllable in self.syllables:
            vowel_count = 0
            for character in syllable:
                if character not in VIETNAMESE_CHARACTERS_LOWER:
                    return False
                if character in VIETNAMESE_VOWELS_LOWER:
                    vowel_count += 1
            if vowel_count > 3:
                return False
            if not split_vietnamese_syllable(syllable):
                return False
        return True

    def split_vietnamese_syllable(self):
        pass


def extract_syllables():
    viwik_file = join(VIWIK_FOLDER, "viwiktionary-20220801-pages-articles-multistream-index.txt")
    syllables = set()
    all_tokens = set()
    with open(viwik_file) as f:
        content = f.read()
        lines = content.splitlines()
    for line in lines:
        tokens = line.split(":")
        if len(tokens) != 3:
            continue
        text = tokens[2]
        for token in text.split(" "):
            all_tokens.add(token)
        word = Word(text)
        if word.is_vietnamese:
            for syllable in word.syllables:
                syllables.add(syllable)
    content = "\n".join(sorted(list(syllables)))
    with open(join("outputs", "syllables.txt"), "w") as f:
        f.write(content)

    content = "\n".join(sorted(list(all_tokens)))
    with open(join("outputs", "all_tokens.txt"), "w") as f:
        f.write(content)


if __name__ == '__main__':
    extract_syllables()
