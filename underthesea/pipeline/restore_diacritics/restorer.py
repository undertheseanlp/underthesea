"""Lightweight Vietnamese diacritics restoration.

Candidate accented syllables are scored with an interpolated syllable
bigram language model and decoded with the Viterbi algorithm. The model
is built from the VNTC news corpus combined with the underthesea
dictionary (see extensions/labs/restore_diacritics/build_model.py).
"""
import gzip
import math
import pickle
import re
from os.path import dirname, join

from underthesea.util.singleton import Singleton
from underthesea.utils.vietnamese_features import remove_tone

MODEL_FILE = "rd_model_2026_08_19.bin"

# Latin letters, including every Vietnamese accented character
WORD_RE = re.compile(r"[a-zA-ZÀ-ɏḀ-ỿ]+")

# punctuation that may wrap a word, e.g. "(hoa" or "sen."
EDGE_PUNCT = "\"'“”‘’.,!?;:()[]{}…-"

# hard punctuation splits words even without surrounding spaces ("mai,Lan")
HARD_PUNCT_RE = r"[,;!?…]+"

# numbers act as context (the <num> token) instead of breaking the chain
NUM_TOKEN = "<num>"
NUM_RE = re.compile(r"[0-9]+(?:[.,:][0-9]+)*")

# weight of the bigram probability against the unigram fallback
LAMBDA = 0.8


def _match_case(token, restored):
    if token.isupper() and len(token) > 1:
        return restored.upper()
    if token[:1].isupper():
        return restored[:1].upper() + restored[1:]
    return restored


@Singleton
class DiacriticsRestorer:
    def __init__(self):
        filepath = join(dirname(__file__), MODEL_FILE)
        with gzip.open(filepath, "rb") as f:
            model = pickle.load(f)
        self.unigrams = model["unigrams"]
        self.bigrams = model["bigrams"]
        variants = {}
        for syllable in self.unigrams:
            variants.setdefault(remove_tone(syllable), set()).add(syllable)
        self.variants = variants
        self.total = sum(self.unigrams.values())
        self.vocab = len(self.unigrams) or 1

    def restore(self, text):
        parts = re.split(r"(\s+|" + HARD_PUNCT_RE + r")", text)
        for chain in self._chains(parts):
            candidates = [
                [NUM_TOKEN] if is_number else self._candidates(core)
                for _, _, core, _, is_number in chain
            ]
            restored = self._viterbi(candidates)
            for (index, prefix, token, suffix, is_number), chosen in zip(
                    chain, restored):
                if not is_number and chosen != token.lower():
                    parts[index] = prefix + _match_case(token, chosen) + suffix
        return "".join(parts)

    def _chains(self, parts):
        """Group words into Viterbi chains.

        Chunks that are neither alphabetic nor numeric (URLs, emails, ...)
        are left untouched and, like punctuation, break the bigram context.
        Numeric chunks are also left untouched but stay in the chain as the
        <num> context token, so e.g. "30 thang 3" reads as a date.
        """
        chains = []
        current = []
        previous_has_suffix = False
        for index, part in enumerate(parts):
            if not part or part.isspace():
                continue
            core = part.strip(EDGE_PUNCT)
            prefix = part[:len(part) - len(part.lstrip(EDGE_PUNCT))]
            suffix = part[len(part.rstrip(EDGE_PUNCT)):]
            if NUM_RE.fullmatch(core):
                is_number = True
            elif core and WORD_RE.fullmatch(core):
                is_number = False
            else:
                if current:
                    chains.append(current)
                    current = []
                previous_has_suffix = False
                continue
            if current and (previous_has_suffix or prefix):
                chains.append(current)
                current = []
            current.append((index, prefix, core, suffix, is_number))
            previous_has_suffix = bool(suffix)
        if current:
            chains.append(current)
        return chains

    def _candidates(self, token):
        lower = token.lower()
        if lower != remove_tone(lower):
            # already contains diacritics, keep as is
            return [lower]
        return sorted(self.variants.get(lower, {lower}))

    def _unigram_prob(self, syllable):
        return (self.unigrams.get(syllable, 0) + 1) / (self.total + self.vocab)

    def _transition(self, previous, current):
        count = self.bigrams.get((previous, current), 0)
        bigram_prob = count / self.unigrams[previous] if count else 0.0
        return math.log(LAMBDA * bigram_prob + (1 - LAMBDA) * self._unigram_prob(current))

    def _viterbi(self, candidates):
        trellis = [{c: (math.log(self._unigram_prob(c)), None) for c in candidates[0]}]
        for i in range(1, len(candidates)):
            layer = {}
            for current in candidates[i]:
                layer[current] = max(
                    (score + self._transition(previous, current), previous)
                    for previous, (score, _) in trellis[-1].items()
                )
            trellis.append(layer)
        best = max(trellis[-1], key=lambda c: trellis[-1][c][0])
        path = [best]
        for layer in reversed(trellis[1:]):
            path.append(layer[path[-1]][1])
        path.reverse()
        return path
