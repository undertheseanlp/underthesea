from collections import OrderedDict
from types import SimpleNamespace

from underthesea.pipeline.word_tokenize.regex_tokenize import VIETNAMESE_VOWELS_LOWER
import regex


class VIETNAMESE:
    LOWERS = SimpleNamespace(
        WITHOUT_DIACRITIC="aăâbcdđeêfghijklmnoôơpqrstuưvwxyz",
        HIGH_LEVEL="aaaeêiooouuy",
        MID_FALLING="àằầèềìòồờùừỳ",
        RISING="áắấéếíóốớúứý",
        LOW_FALLING_RISING="ảẳẩẻểỉỏổởủửỷ",
        HIGH_FALLING_RISING_GLOTTALIZED="ãẵẫẽễĩõỗỡũữỹ",
        LOW_GLOTTALIZED="ạặậẹệịọộợụựỵ"
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
    def remove_tone(cls, s):
        pass


VIETNAMESE_LOWERS_MID_FALLING_TO_HIGH_LEVEL = dict(
    zip(VIETNAMESE.LOWERS.MID_FALLING, VIETNAMESE.LOWERS.HIGH_LEVEL))
VIETNAMESE_LOWERS_RISING_TO_HIGH_LEVEL = dict(
    zip(VIETNAMESE.LOWERS.RISING, VIETNAMESE.LOWERS.HIGH_LEVEL))
VIETNAMESE_LOWERS_LOW_FALLING_RISING_TO_HIGH_LEVEL = dict(
    zip(VIETNAMESE.LOWERS.LOW_FALLING_RISING, VIETNAMESE.LOWERS.HIGH_LEVEL))
VIETNAMESE_LOWERS_HIGH_FALLING_RISING_GLOTTALIZED_TO_HIGH_LEVEL = dict(
    zip(VIETNAMESE.LOWERS.HIGH_FALLING_RISING_GLOTTALIZED, VIETNAMESE.LOWERS.HIGH_LEVEL))
VIETNAMESE_LOWERS_LOW_GLOTTALIZED_TO_HIGH_LEVEL = dict(
    zip(VIETNAMESE.LOWERS.LOW_GLOTTALIZED, VIETNAMESE.LOWERS.HIGH_LEVEL))


class Syllable:
    def __init__(self, text):
        self.text = text
        i = "[iìíỉĩị]"
        y = "[yỳýỷỹỵ]"
        a = "[aàáảãạăằắẳẵặâầấẩẫậ]"
        a_circumflex = "[âầấẩẫậ]"
        e = "[êềếểễệ]"
        e_circumflex = "[êềếểễệ]"
        o = "[oòôôồốổỗộơờớởỡợ]"
        u = "[uùúủũụưừứửữự]"
        double = f"{i}{a}|{i}{e}|{y}{e}|{y}{a}|{u}{o}|{u}{a}"
        v = r"(?P<V>[" + VIETNAMESE_VOWELS_LOWER + "]|" + double + ")"
        vy = r"(?P<V>" + y + ")"
        vac = r"(?P<V>" + a_circumflex + ")"
        vec = r"(?P<V>" + e_circumflex + ")"
        G = r"(?P<G>[iyou])?"
        w = r"(?P<w>[o])?"
        wo = r"(?P<w>[oòóỏõọ])?"
        wu = r"(?P<w>[uù])?"
        Vye = f"(?P<V>y{e})"
        Vya = f"(?P<V>y{a})"
        consonants = "gi|qu|ch|gh|kh|ng|ngh|nh|ph|th|tr|[bcdđghklmnpqrstvx]"
        c1 = r"(?P<C1>" + consonants + ")?"
        c2 = r"(?P<C2>" + consonants + ")?"
        patterns = [
            r"^" + c1 + w + v + G + c2 + "$",
            r"^" + c1 + wu + vac + G + c2 + "$",
            r"^" + c1 + wo + r"(?P<V>[ea])" + G + c2 + "$",
            r"^" + c1 + wu + vec + G + c2 + "$",
            r"^" + c1 + wu + vy + G + c2 + "$",
            r"^" + c1 + wu + Vye + G + c2 + "$",
            r"^" + c1 + wu + Vya + G + c2 + "$",
            r"^" + c1 + "y" + "$"
        ]
        pattern = r"(" + "|".join(patterns) + ")"

        matched = regex.match(pattern, text)
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

        Args:
            tone (str): ipa or number
        """
        groups = self.matched.groupdict()
        non_diacritic = {
            "a": ["a", "à", "á", "ả", "ã", "ạ"],
            "ă": ["ă", "ằ", "ắ", "ẳ", "ẵ", "ặ"],
            "â": ["â", "ầ", "ấ", "ẩ", "ẫ", "ậ"],
            "e": ["e", "è", "é", "ẻ", "ẽ", "ẹ"],
            "ê": ["ê", "ề", "ế", "ể", "ễ", "ệ"],
            "i": ["i", "ì", "í", "ỉ", "ĩ", "ị"],
            "ia": ["ia", "ìa", "ía", "ỉa", "ĩa", "ịa"],
            "iê": ["iê", "iề", "iế", "iể", "iễ", "iệ"],
            "o": ["o", "ò", "ó", "ỏ", "õ", "ọ"],
            "ô": ["ô", "ồ", "ố", "ổ", "ỗ", "ộ"],
            "ơ": ["ơ", "ờ", "ớ", "ở", "ỡ", "ợ"],
            "u": ["u", "ù", "ú", "ủ", "ũ", "ụ"],
            "ua": ["ua", "ùa", "úa", "ủa", "ũa", "ụa"],
            "uâ": ["uâ", "uầ", "uấ", "uẩ", "uẫ", "uậ"],
            "uô": ["uô", "uồ", "uố", "uổ", "uỗ", "uộ"],
            "ư": ["ư", "ừ", "ứ", "ử", "ữ", "ự"],
            "ưa": ["ưa", "ừa", "ứa", "ửa", "ữa", "ựa"],
            "ươ": ["ươ", "ườ", "ướ", "ưở", "ưỡ", "ượ"],
            "y": ["y", "ỳ", "ý", "ỷ", "ỹ", "ỵ"],
            "yê": ["yê", "yề", "yế", "yể", "yễ", "yệ"],
            "ya": ["ya"]
        }
        non_diacritic = self._util_reverse_dict(non_diacritic)
        map_V = {
            "a": ["a", "à", "á", "ả", "ã", "ạ"],
            "ă": ["ă", "ằ", "ắ", "ẳ", "ẵ", "ặ"],
            "ɤ̆": ["â", "ầ", "ấ", "ẩ", "ẫ", "ậ"],
            "i": [
                "i", "ì", "í", "ỉ", "ĩ", "ị",
                "y", "ỳ", "ý", "ỷ", "ỹ", "ỵ"
            ],
            "ɨ": ["ư", "ừ", "ứ", "ử", "ữ", "ự"],
            "ia": ["ia", "ìa"],
            "ɛ": ["e", "è", "é", "ẻ", "ẽ", "ẹ", "ê", "ề", "ế", "ể", "ễ", "ệ"],
            "ɔ": ["o", "ò", "ó", "ỏ", "õ", "ọ"],
            "o": ["ô", "ồ", "ố", "ổ", "ỗ", "ộ"],
            "əː": ["ơ", "ờ", "ớ", "ở", "ỡ", "ợ"],
            "u": ["u", "ù", "ú", "ủ", "ũ", "ụ"],
            "iə": [
                "iê", "iề", "iế", "iể", "iễ", "iệ",
                "ìa", "ía", "ỉa", "ĩa", "ịa",
                "yê", "yề", "yế", "yể", "yễ", "yệ",
                "ya",
            ],
            "ɨə̰": [
                "ươ", "ườ", "ướ", "ưở", "ưỡ", "ượ",
                "ưa", "ừa", "ứa", "ửa", "ữa", "ựa"
            ],
            "wə": [
                "uâ", "uầ", "uấ", "uẩ", "uẫ", "uậ"
            ],
            "uə": [
                "uô", "uồ", "uố", "uổ", "uỗ", "uộ",
                "ua", "ùa", "úa", "ủa", "ũa", "ụa"
            ],
        }
        map_V = self._util_reverse_dict(map_V)

        map_C = {
            # "ɓ": ["b"],
            "b": ["b"],
            "k": ["c", "k", "q"],
            "kʰ": ["kh"],
            "ʨ": ["ch", "tr"],
            "z": ["d", "gi"],
            "ɣ": ["g", "gh"],
            "h": ["h"],
            "l": ["l"],
            "m": ["m"],
            "n": ["n"],
            "ŋ": ["ng", "ngh"],
            "ɲ": ["nh"],
            "p": ["p"],
            "f": ["ph"],
            "kw": ["qu"],
            "r": ["r"],
            "s": ["s", "x"],
            "t": ["t"],
            "tʰ": ["th"],
            "v": ["v"],
            "ɗ": ["đ"],
        }
        map_C = self._util_reverse_dict(map_C)

        map_w = {
            "w": ["o", "ò", "ó", "ỏ", "õ", "ọ", "u"]
        }
        map_w = self._util_reverse_dict(map_w)

        ipa = ""

        if groups['C1']:
            C1 = map_C[groups['C1']]
            ipa += C1

        if groups['w']:
            w = map_w[groups['w']]
            ipa += w

        V = non_diacritic[groups['V']]
        if groups['G']:
            G = groups['G']
            if G == 'o':
                if V in ["a", "à", "á", "ả", "ã", "ạ"]:
                    ipa += 'aw'
                elif V in ["e", "è", "é", "ẻ", "ẽ", "ẹ"]:
                    ipa += 'ɛw'
                else:
                    raise Exception('Need implement')
            elif G == 'i':
                ipa += map_V[V] + "j"
            elif G == 'u':
                if V in ["a", "à", "á", "ả", "ã", "ạ"]:
                    ipa += 'ăw'
                elif V == 'ê':
                    ipa += 'ew'
                elif V == 'i':
                    ipa += 'iw'
                elif V == 'ư':
                    ipa += 'ɨw'
                elif V in ["â", "iê", "ươ", "yê"]:
                    ipa += map_V[V] + "w"
                elif V == 'y':
                    ipa += "iw"
                else:
                    raise Exception('Need implement')
            elif G == 'y':
                if V in ["a", "à", "á", "ả", "ã", "ạ"]:
                    ipa += 'ăj'
                elif V == 'u':
                    ipa += 'wi'
                elif V in ["â", "uâ"]:
                    ipa += map_V[V] + 'j'
                else:
                    raise Exception('Need implement')
            else:
                raise Exception('Need implement')
        else:
            V = map_V[V]
            ipa += V

        if groups['C2']:
            C2 = map_C[groups['C2']]
            ipa += C2

        T = VIETNAMESE.TONE.HIGH_LEVEL
        for character in self.text:
            if character in VIETNAMESE.LOWERS.MID_FALLING:
                T = VIETNAMESE.TONE.MID_FALLING
            elif character in "áắấéếíóớốúứý":
                T = VIETNAMESE.TONE.RISING
            elif character in "ảẳẩẻểỉỏởổủửỷ":
                T = VIETNAMESE.TONE.LOW_FALLING_RISING
            elif character in "ãẵẫẽễĩõỡỗũữỹ":
                T = VIETNAMESE.TONE.HIGH_FALLING_RISING_GLOTTALIZED
            elif character in "ạặậẹệịọợộụựỵ":
                T = VIETNAMESE.TONE.LOW_GLOTTALIZED

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
        T = map_T[T]
        ipa += T
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
