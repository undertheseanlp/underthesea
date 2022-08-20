from underthesea.pipeline.word_tokenize.regex_tokenize import VIETNAMESE_VOWELS_LOWER
import regex


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

        map_G = {
            "j": ["i"],
            "w": ["o", "u", "y"]
        }
        map_G = self._util_reverse_dict(map_G)

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

        T = "high_level"
        for character in self.text:
            if character in "àằầèềìòờồùừỳ":
                T = "mid_falling"  # `
            elif character in "áắấéếíóớốúứý":
                T = "rising"  # /
            elif character in "ảẳẩẻểỉỏởổủửỷ":
                T = "low_falling_rising"  # ?
            elif character in "ãẵẫẽễĩõỡỗũữỹ":
                T = "high_falling_rising_glottalized"  # ~
            elif character in "ạặậẹệịọợộụựỵ":
                T = "low_glottalized"  # .

        map_tone_ipa = {
            "high_level": "˧˧",
            "mid_falling": "˨˩",
            "rising": "˨˩",
            "low_glottalized": "˧˥"
        }

        map_tone_number = {
            "high_level": "³³",
            "mid_falling": "²¹",
            "rising": "³⁴",
            "low_falling_rising": "³⁰¹",
            "high_falling_rising_glottalized": "³¹ˀ⁵",
            "low_glottalized": "¹¹ˀ"
        }
        if tone == "number":
            map_T = map_tone_number
        else:
            map_T = map_tone_ipa
        T = map_T[T]
        ipa += T
        return ipa
