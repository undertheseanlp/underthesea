from collections import OrderedDict
from types import SimpleNamespace

from underthesea.pipeline.word_tokenize.regex_tokenize import VIETNAMESE_VOWELS_LOWER
import regex


class VIETNAMESE:
    LOWERS = SimpleNamespace(
        WITHOUT_DIACRITIC="aăâbcdđeêfghijklmnoôơpqrstuưvwxyz",
        HIGH_LEVEL="aăâeêioôơuưy",
        MID_FALLING="àằầèềìòồờùừỳ",
        RISING="áắấéếíóốớúứý",
        LOW_FALLING_RISING="ảẳẩẻểỉỏổởủửỷ",
        HIGH_FALLING_RISING_GLOTTALIZED="ãẵẫẽễĩõỗỡũữỹ",
        LOW_GLOTTALIZED="ạặậẹệịọộợụựỵ",
    )

    TONE = SimpleNamespace(
        HIGH_LEVEL="HIGH_LEVEL",  # e.g. ta, ba
        MID_FALLING="MID_FALLING",  # e.g. tà, bà
        RISING="RISING",  # e.g. tá, bá
        LOW_FALLING_RISING="LOW_FALLING_RISING",  # e.g. tả, bả
        HIGH_FALLING_RISING_GLOTTALIZED="HIGH_FALLING_RISING_GLOTTALIZED",  # e.g. tã, bã
        LOW_GLOTTALIZED="LOW_GLOTTALIZED"  # e.g. tạ, bạ
    )

    @classmethod
    def analyze_tone(cls, s):
        hl = cls.LOWERS.HIGH_LEVEL
        mf = cls.LOWERS.MID_FALLING
        r = cls.LOWERS.RISING
        lfr = cls.LOWERS.LOW_FALLING_RISING
        hfrg = cls.LOWERS.HIGH_FALLING_RISING_GLOTTALIZED
        lg = cls.LOWERS.LOW_GLOTTALIZED
        d1 = mf + r + lfr + hfrg + lg
        d2 = hl + hl + hl + hl + hl
        tmp = s.maketrans(d1, d2)
        non_tone_letters = s.translate(tmp)

        tone = cls.TONE.HIGH_LEVEL
        for c in s:
            if c in mf:
                tone = cls.TONE.MID_FALLING
            elif c in r:
                tone = cls.TONE.RISING
            elif c in lfr:
                tone = cls.TONE.LOW_FALLING_RISING
            elif c in hfrg:
                tone = cls.TONE.HIGH_FALLING_RISING_GLOTTALIZED
            elif c in lg:
                tone = cls.TONE.LOW_GLOTTALIZED
        return non_tone_letters, tone


class Syllable:
    def __init__(self, text):
        self.text = text
        non_tone_letters, tone = VIETNAMESE.analyze_tone(text)
        self.tone = tone
        i = "i"
        y = "y"
        a = "[aăâ]"
        a_circumflex = "â"
        e = "[ê]"
        o = "[oôơ]"
        u = "[uư]"
        double = f"oo|{i}{a}|{i}{e}|{y}{e}|{y}{a}|{u}{o}|{u}{a}"
        v = r"(?P<V>" + double + "|[" + VIETNAMESE_VOWELS_LOWER + "])"
        vy = r"(?P<V>" + y + ")"
        vac = r"(?P<V>" + a_circumflex + ")"
        vec = r"(?P<V>" + e + ")"
        G = r"(?P<G>[iyou])?"
        w = r"(?P<w>[o])?"
        wo = r"(?P<w>[o])?"
        wu = r"(?P<w>[u])?"
        Vye = f"(?P<V>y{e})"
        Vya = f"(?P<V>y{a})"
        consonants = "gi|qu|ch|gh|kh|ng|ngh|nh|ph|th|tr|[bcdđghklmnpqrstvx]"
        c1 = r"(?P<C1>" + consonants + ")?"
        c2 = r"(?P<C2>" + consonants + ")?"
        patterns = [
            r"^" + c1 + wu + vy + G + c2 + "$",
            r"^" + c1 + v + G + c2 + "$",
            r"^" + c1 + w + v + G + c2 + "$",
            r"^" + c1 + wu + vac + G + c2 + "$",
            r"^" + c1 + wo + r"(?P<V>[ea])" + G + c2 + "$",
            r"^" + c1 + wu + vec + G + c2 + "$",
            r"^" + c1 + wu + Vye + G + c2 + "$",
            r"^" + c1 + wu + Vya + G + c2 + "$"
        ]
        pattern = r"(" + "|".join(patterns) + ")"

        matched = regex.match(pattern, non_tone_letters)
        self.matched = matched
        if not matched:
            raise Exception(f"Text {text} not matched")

    def _util_reverse_dict(self, d):
        result = {}
        for k in d:
            for v in d[k]:
                result[v] = k
        return result

    # flake8: noqa: C901
    def generate_ipa(self, tone='number'):
        """
        syllable = c1 + (w) + v + G + c2
        G: iyou
        TODO: merge G with c2

        Args:
            tone (str): ipa or number
        """
        groups = self.matched.groupdict()

        map_V = {
            "a": ["a"],
            "ă": ["ă"],
            "ɤ̆": ["â"],
            "i": ["i", "y"],
            "ɯ": ["ư"],
            "ɛ": ["e"],
            "e": ["ê"],
            "ↄ": ["o"],
            "o": ["ô"],
            "ɤ": ["ơ"],
            "u": ["u"],
            "iə": ["ie", "iê", "ia", "yê", "ya"],
            "ɯə̰": ["ưa"],
            "ɯə": ["ươ"],
            "wə": ["uâ"],
            "uə": ["uô", "ua"],
            "ↄ:": ["oo"]
        }
        map_V = self._util_reverse_dict(map_V)

        map_C = {
            "b": ["b"],
            "k": ["c", "k", "q"],
            "kʰ": ["kh"],
            "ʨ": ["ch", "tr"],
            "z": ["d"],
            "zi": ["gi"],
            "ɣ": ["g", "gh"],
            "h": ["h"],
            "l": ["l"],
            "m": ["m"],
            "n": ["n"],
            "ŋ": ["ng", "ngh", "nh"],
            "p": ["p"],
            "f": ["ph"],
            "kw": ["qu"],
            "r": ["r"],
            "s": ["s"],
            "x": ["x"],
            "t": ["t"],
            "tʰ": ["th"],
            "v": ["v"],
            "ɗ": ["đ"],
        }

        map_C2 = {
            "b": ["b"],
            "k": ["c", "k", "q", "ch"],
            "kʰ": ["kh"],
            "z": ["d", "gi"],
            "ɣ": ["g", "gh"],
            "h": ["h"],
            "l": ["l"],
            "m": ["m"],
            "n": ["n"],
            "ŋ": ["ng", "ngh", "nh"],
            "p": ["p"],
            "f": ["ph"],
            "kw": ["qu"],
            "r": ["r"],
            "s": ["s"],
            "x": ["x"],
            "t": ["t"],
            "tʰ": ["th"],
            "v": ["v"],
            "ɗ": ["đ"],
        }
        map_C = self._util_reverse_dict(map_C)
        map_C2 = self._util_reverse_dict(map_C2)

        map_w = {
            "ʷ": ["o", "u"]
        }
        map_w = self._util_reverse_dict(map_w)

        map_G = {
            "j": ["i", "y"],
            "w": ["o", "u"]
        }
        map_G = self._util_reverse_dict(map_G)

        C1, w, V, G, C2 = groups['C1'], groups['w'], groups['V'], groups['G'], groups['C2']

        if w:
            ipa_w = map_w[groups['w']]
        else:
            ipa_w = ""

        ipa_V = map_V[V]
        if C1:
            ipa_C1 = map_C[C1]
        else:
            ipa_C1 = ""
            if V == "a":
                ipa_V = "ɛ̆"

        if ipa_C1 == "":
            ipa_C1 = "ʔ"

        if G:
            ipa_G = map_G[G]
            if V == "a":
                if G == "o":
                    # This rule apply in case ao -> ʔaw³³ not ɛ̆w³
                    ipa_V = "a"
                elif G in ["u", "ă", "y"]:
                    ipa_V = "ă"
            if V == "o" and G == "o":
                ipa_V = "ↄ:"
                ipa_G = ""
            if V == "y" and G == "u":
                ipa_V = "i"
            if V == "u" and G == "y":
                ipa_G = "i"
        else:
            ipa_G = ""

        if C2:
            ipa_C2 = map_C2[C2]
        else:
            ipa_C2 = ""

        map_tone_ipa = {
            VIETNAMESE.TONE.HIGH_LEVEL: "˧˧",
            VIETNAMESE.TONE.MID_FALLING: "˨˩",
            VIETNAMESE.TONE.LOW_FALLING_RISING: "˨˩",
            VIETNAMESE.TONE.HIGH_FALLING_RISING_GLOTTALIZED: "˧˥"
        }

        map_tone_number = {
            VIETNAMESE.TONE.HIGH_LEVEL: "³³",
            VIETNAMESE.TONE.MID_FALLING: "²¹",
            VIETNAMESE.TONE.RISING: "³⁴",
            VIETNAMESE.TONE.LOW_FALLING_RISING: "³⁰¹",
            VIETNAMESE.TONE.HIGH_FALLING_RISING_GLOTTALIZED: "³¹ˀ⁵",
            VIETNAMESE.TONE.LOW_GLOTTALIZED: "¹¹ˀ"
        }
        if tone == "number":
            map_T = map_tone_number
        else:
            map_T = map_tone_ipa
        ipa_T = map_T[self.tone]

        if ipa_C2 in ["p", "t", "k"]:
            if self.tone == VIETNAMESE.TONE.RISING:
                ipa_T = "⁴⁵"
            elif self.tone == VIETNAMESE.TONE.LOW_GLOTTALIZED:
                ipa_T = "¹⁰ˀ"
        if ipa_C2 in ["ŋ", "k"] and ipa_V in ["u", "o", "ↄ"]:
            if ipa_C2 == "ŋ":
                ipa_C2 = "ŋ͡m"
            elif ipa_C2 == "k":
                ipa_C2 = "k͡p"
        ipa = ipa_C1 + ipa_w + ipa_V + ipa_G + ipa_C2 + ipa_T
        return ipa


vietnamese_alphabet = "aàáảãạ" \
                      + "ăằắẳẵặ" \
                      + "âầấẩẫậ" \
                      + "bcd" \
                      + "đ" \
                      + "eèéẻẽẹ" \
                      + "êềếểễệ" \
                      + "fgh" \
                      + "iìíỉĩị" \
                      + "jklmn" \
                      + "oòóỏõọ" \
                      + "ôồốổỗộ" \
                      + "ơờớởỡợ" \
                      + "pqrst" \
                      + "uùúủũụ" \
                      + "ưừứửữự" \
                      + "vwx" \
                      + "yỳýỷỹỵ" \
                      + "z"

vietnamese_alphabet_order = OrderedDict()
for i, c in enumerate(vietnamese_alphabet):
    vietnamese_alphabet_order[c] = i


def vietnamese_sort_key(s):
    return [vietnamese_alphabet_order[c] for c in s]
