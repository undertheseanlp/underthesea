"""
Underthesea Vietnamese IPA
"""
from collections import OrderedDict
from types import SimpleNamespace
import regex
from underthesea.pipeline.word_tokenize.regex_tokenize import VIETNAMESE_VOWELS_LOWER
from underthesea.utils.vietnamese_ipa_rules import codas, nuclei, onsets


class VIETNAMESE:
    """VIETNAMESE NAMESPACES"""

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
        LOW_GLOTTALIZED="LOW_GLOTTALIZED",  # e.g. tạ, bạ
    )

    @classmethod
    def analyze_tone(cls, text: str):
        """

        Args:
            text (str): a syllable
        """
        hl = cls.LOWERS.HIGH_LEVEL
        mf = cls.LOWERS.MID_FALLING
        r = cls.LOWERS.RISING
        lfr = cls.LOWERS.LOW_FALLING_RISING
        hfrg = cls.LOWERS.HIGH_FALLING_RISING_GLOTTALIZED
        lg = cls.LOWERS.LOW_GLOTTALIZED
        d1 = mf + r + lfr + hfrg + lg
        d2 = hl + hl + hl + hl + hl
        tmp = text.maketrans(d1, d2)
        non_tone_letters = text.translate(tmp)

        tone = cls.TONE.HIGH_LEVEL
        for c in text:
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
    """
    Syllable class
    """

    def __init__(self, text):
        self.text = text
        non_tone_letters, tone = VIETNAMESE.analyze_tone(text)
        self.tone = tone
        a = "[aăâ]"
        o = "[oôơ]"
        u = "[uư]"
        double = f"oo|i{a}|iê|yê|y{a}|{u}{o}|{u}{a}"
        v = r"(?P<V>" + double + "|[" + VIETNAMESE_VOWELS_LOWER + "]|y)"
        # vy = r"(?P<V>y)"
        vac = r"(?P<V>â)"
        vec = r"(?P<V>ê)"
        wu = r"(?P<w>[u])"
        Vye = "(?P<V>yê)"
        Vya = f"(?P<V>y{a})"
        consonants = "gi|qu|ch|gh|kh|ng|ngh|nh|ph|th|tr|[bcdđghklmnpqrstvx]"
        conda = consonants + "|[uiyo]"
        c1 = r"(?P<C1>" + consonants + ")?"
        c2 = r"(?P<C2>" + conda + ")?"
        patterns = [
            r"^" + c1 + "(?P<w>[u])(?P<V>[yâ])" + c2 + "$",
            r"^" + c1 + "(?P<w>[o])(?P<V>[eaă])" + c2 + "$",
            r"^" + c1 + v + c2 + "$",
            r"^" + c1 + wu + vac + c2 + "$",
            r"^" + c1 + wu + vec + c2 + "$",
            r"^" + c1 + wu + Vye + c2 + "$",
            r"^" + c1 + wu + Vya + c2 + "$",
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
    def generate_ipa(
        self, dialect: str = "north", eight: bool = False, tone: str = "number"
    ):
        """Generate ipa of the syllable

        Syllable structure
        syllable = onset + (w) + nuclei + conda

        Args:
            dialect (str): Either the `string` `"north"` or `"south"`
            eight (boolean): If true, use eight tone format, else use six tone format
            tone (str): Either the `string` `"ipa"` or `"number"`

        Returns:
            A `string`. Represents ipa of the syllable
        """
        groups = self.matched.groupdict()

        map_w = {"ʷ": ["o", "u"]}
        map_w = self._util_reverse_dict(map_w)

        C1, w, V, C2 = groups["C1"], groups["w"], groups["V"], groups["C2"]

        if w:
            ipa_w = map_w[groups["w"]]
        else:
            ipa_w = ""

        ipa_V = nuclei[V]
        if C1:
            ipa_C1 = onsets[C1]
        else:
            ipa_C1 = ""
            if V == "a":
                # ipa_V = "ɛ̆"
                ipa_V = "aː"

        if ipa_C1 == "":
            # ipa_C1 = "ʔ"
            ipa_C1 = ""

        if C2:
            ipa_C2 = codas[C2]
            if C2 in ["o", "i", "u", "y"]:
                if V == "a":
                    if C2 == "o":
                        # This rule apply in case ao -> ʔaw³³ not ɛ̆w³
                        # ipa_V = "a"
                        ipa_V = "aː"
                    elif C2 in ["u", "ă", "y"]:
                        # trang
                        # ipa_V = "ă"
                        # vphon
                        ipa_V = "a"
                if V == "o" and C2 == "o":
                    ipa_V = "ↄ:"
                    ipa_C2 = ""
                if V == "y" and C2 == "u":
                    ipa_V = "i"
                if V == "u" and C2 == "y":
                    ipa_C2 = "i"
        else:
            ipa_C2 = ""

        map_tone_ipa = {
            VIETNAMESE.TONE.HIGH_LEVEL: "˧˧",
            VIETNAMESE.TONE.MID_FALLING: "˨˩",
            VIETNAMESE.TONE.LOW_FALLING_RISING: "˨˩",
            VIETNAMESE.TONE.HIGH_FALLING_RISING_GLOTTALIZED: "˧˥",
        }

        # vphone
        map_tone_number = {
            VIETNAMESE.TONE.HIGH_LEVEL: "³³",
            VIETNAMESE.TONE.MID_FALLING: "³²",
            VIETNAMESE.TONE.RISING: "²⁴",
            VIETNAMESE.TONE.LOW_FALLING_RISING: "³¹²",
            VIETNAMESE.TONE.HIGH_FALLING_RISING_GLOTTALIZED: "³ˀ⁵",
            VIETNAMESE.TONE.LOW_GLOTTALIZED: "²¹ˀ",
        }
        if tone == "number":
            map_T = map_tone_number
        else:
            map_T = map_tone_ipa
        ipa_T = map_T[self.tone]

        if eight:
            if ipa_C2 in ["p", "t", "k"]:
                if self.tone == VIETNAMESE.TONE.RISING:
                    ipa_T = "⁴⁵"
                elif self.tone == VIETNAMESE.TONE.LOW_GLOTTALIZED:
                    ipa_T = "²¹"
        if ipa_C2 in ["ŋ", "k"] and ipa_V in ["u", "o", "ↄ"]:
            if ipa_C2 == "ŋ":
                ipa_C2 = "ŋ͡m"
            elif ipa_C2 == "k":
                ipa_C2 = "k͡p"

        ons = ipa_C1
        cod = ipa_C2
        nuc = ipa_V

        if ons == "":
            ons = "ʔ"
        if ons == "ʂ":
            ons = "s"

        # Capture vowel/coda interactions of ɛ/ɛː and e/eː
        if cod in ["ŋ", "k"]:
            if nuc == "ɛ":
                nuc = "ɛː"
            if nuc == "e":
                nuc = "eː"

        # Velar fronting
        if nuc == "aː":
            if cod == "c":
                nuc = "ɛ"
            if cod == "ɲ":
                nuc = "ɛ"

        # No surface palatal codas
        if cod in ["c", "ɲ"]:
            if cod == "c":
                cod = "k"
            if cod == "ɲ":
                cod = "ŋ"

        if not cod and nuc in ["aː", "əː"]:
            if nuc == "aː":
                nuc = "a"
            if nuc == "əː":
                nuc = "ə"
        ipa = ons + ipa_w + nuc + cod + ipa_T
        return ipa


vietnamese_alphabet = (
    "aàáảãạ"
    + "ăằắẳẵặ"
    + "âầấẩẫậ"
    + "bcd"
    + "đ"
    + "eèéẻẽẹ"
    + "êềếểễệ"
    + "fgh"
    + "iìíỉĩị"
    + "jklmn"
    + "oòóỏõọ"
    + "ôồốổỗộ"
    + "ơờớởỡợ"
    + "pqrst"
    + "uùúủũụ"
    + "ưừứửữự"
    + "vwx"
    + "yỳýỷỹỵ"
    + "z"
)

vietnamese_alphabet_order = OrderedDict()
for i, character in enumerate(vietnamese_alphabet):
    vietnamese_alphabet_order[character] = i


def vietnamese_sort_key(s):
    return [vietnamese_alphabet_order[c] for c in s]
