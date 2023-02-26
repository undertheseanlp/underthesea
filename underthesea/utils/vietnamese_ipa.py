"""
Underthesea Vietnamese IPA
"""
from collections import OrderedDict
from types import SimpleNamespace
import regex
from underthesea.pipeline.word_tokenize.regex_tokenize import VIETNAMESE_VOWELS_LOWER
from underthesea.utils.vietnamese_ipa_rules import codas, nuclei, onsets, onoffglides, onglides, offglides


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
        double = f"oo|i{a}|iê|yê|y{a}|{u}{o}|{u}{a}|ay|ây|ua"
        v = r"(?P<V>[aăâeêuưyoôơi]|" + double + ")"
        vec = r"(?P<V>ê)"
        wu = r"(?P<w>[u])"
        vye = "(?P<V>yê)"
        vya = f"(?P<V>y{a})"
        consonants = "gi|qu|ch|gh|kh|ng|ngh|nh|ph|th|tr|[bcdđghklmnpqrstvx]"
        conda = consonants + "|[uio]"
        c1 = r"(?P<C1>" + consonants + ")?"
        c2 = r"(?P<C2>" + conda + ")?"
        patterns = [
            r"^" + c1 + "(?P<w>[u])(?P<V>[yâ])" + c2 + "$",
            r"^" + c1 + "(?P<w>[o])(?P<V>[eaă])" + c2 + "$",
            r"^" + c1 + v + c2 + "$",
            r"^" + c1 + wu + vec + c2 + "$",
            r"^" + c1 + wu + vye + c2 + "$",
            r"^" + c1 + wu + vya + c2 + "$",
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

        Vietnamese syllabic structure (Trang 2022)
        syllable = onset + rhyme + tone
        rhyme = medial + nuclear vowel + (coda)

        Args:
            dialect (str): Either the `string` `"north"` or `"south"`. Default: `north`
            eight (boolean): If true, use eight tone format, else use six tone format. Default: `False`
            tone (str): Either the `string` `"ipa"` or `"number"`. Default: `number`

        Returns:
            A `string`. Represents ipa of the syllable
        """
        groups = self.matched.groupdict()

        map_w = {
            "o": "ʷ",
            "u": "ʷ"
        }

        c1, w, v, c2 = groups["C1"], groups["w"], groups["V"], groups["C2"]

        ipa_w = ""
        if w:
            ipa_w = map_w[w]

        nuclei.update(onglides)
        nuclei.update(offglides)
        nuclei.update(onoffglides)
        ipa_v = nuclei[v]
        if c1:
            ipa_c1 = onsets[c1]
        else:
            ipa_c1 = ""
            if v == "a":
                # ipa_v = "ɛ̆"
                ipa_v = "aː"

        if ipa_c1 == "":
            # ipa_c1 = "ʔ"
            ipa_c1 = ""

        if c2:
            ipa_c2 = codas[c2]
            if c2 in ["o", "i", "u", "y"]:
                if v == "a":
                    if c2 == "o":
                        # This rule apply in case ao -> ʔaw³³ not ɛ̆w³
                        # ipa_v = "a"
                        ipa_v = "aː"
                    elif c2 in ["u", "ă", "y"]:
                        ipa_v = "a"
                if v == "o" and c2 == "o":
                    ipa_v = "ↄ:"
                    ipa_c2 = ""
                if v == "y" and c2 == "u":
                    ipa_v = "i"
                if v == "u" and c2 == "y":
                    ipa_c2 = "i"
        else:
            ipa_c2 = ""

        map_tone_ipa = {
            VIETNAMESE.TONE.HIGH_LEVEL: "˧˧",
            VIETNAMESE.TONE.MID_FALLING: "˨˩",
            VIETNAMESE.TONE.LOW_FALLING_RISING: "˨˩",
            VIETNAMESE.TONE.HIGH_FALLING_RISING_GLOTTALIZED: "˧˥",
        }

        # vphon
        map_tone_number = {
            VIETNAMESE.TONE.HIGH_LEVEL: "³³",
            VIETNAMESE.TONE.MID_FALLING: "³²",
            VIETNAMESE.TONE.RISING: "²⁴",
            VIETNAMESE.TONE.LOW_FALLING_RISING: "³¹²",
            VIETNAMESE.TONE.HIGH_FALLING_RISING_GLOTTALIZED: "³ˀ⁵",
            VIETNAMESE.TONE.LOW_GLOTTALIZED: "²¹ˀ",
        }
        if tone == "number":
            map_t = map_tone_number
        else:
            map_t = map_tone_ipa
        ipa_t = map_t[self.tone]

        ons = ipa_c1
        cod = ipa_c2
        nuc = ipa_v
        ton = ipa_t

        # Deal with gi, giền and giêng
        if ons == "z" and nuc == "e":
            nuc = "iə"
        if ons == "ɣ" and nuc == "i":
            ons = "z"

        ##
        # Generate internal G2P representation
        ##
        if ons == "":
            ons = "ʔ"
        if ons == "ʂ":
            ons = "s"

        ##
        # Northern
        ##
        if dialect == "north":
            # Onset mergers
            if ons in ["j", "r"]:
                ons = "z"
            elif ons in ["c", "ʈ"]:
                ons = "tɕ"
            elif ons == "ʂ":
                ons = "s"

        if dialect == "south":
            if cod in ["ŋ", "k"] and nuc in ["u", "o", "ↄ"]:
                if cod == "ŋ":
                    cod = "ŋ͡m"
                elif cod == "k":
                    cod = "k͡p"

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

        if eight:
            if cod in ["p", "t", "k"]:
                if self.tone == VIETNAMESE.TONE.RISING:
                    ton = "⁴⁵"
                elif self.tone == VIETNAMESE.TONE.LOW_GLOTTALIZED:
                    ton = "²¹"
        ipa = ons + ipa_w + nuc + cod + ton
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
