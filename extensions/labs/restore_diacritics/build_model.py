"""Build the syllable n-gram model for underthesea.pipeline.restore_diacritics.

The model combines two sources:
  * VNTC news corpus (train split) — real-world syllable statistics
  * underthesea dictionary — headwords (weighted x3), definitions, examples

Counts are pruned (unigram >= 5, bigram >= 5) and stored as a gzipped
pickle: {"unigrams": {syllable: count}, "bigrams": {(s1, s2): count}}.

Usage (from repo root):
    python -c "from underthesea.data_fetcher import DataFetcher; \\
               DataFetcher.download_data('VNTC', None)"
    python extensions/labs/restore_diacritics/build_model.py
"""
import gzip
import pickle
import re
from collections import Counter
from os.path import dirname, expanduser, join

from underthesea.dictionary import Dictionary
from underthesea.pipeline.restore_diacritics.restorer import (
    MODEL_FILE,
    NUM_TOKEN,
    WORD_RE,
)

# words plus digit runs; digit runs are counted as the <num> context token
TOKEN_RE = re.compile(WORD_RE.pattern + r"|[0-9]+")

UNIGRAM_MIN_COUNT = 5
BIGRAM_MIN_COUNT = 5
HEADWORD_WEIGHT = 3

# Vietnamese syllables never contain these letters and are short
INVALID_LETTERS = set("fjwzFJWZ")

VNTC_TRAIN = expanduser("~/.underthesea/datasets/VNTC/train.txt")
OUTPUT = join(dirname(__file__), "..", "..", "..",
              "underthesea", "pipeline", "restore_diacritics", MODEL_FILE)


def valid_syllable(syllable):
    if syllable == NUM_TOKEN:
        return True
    return len(syllable) <= 7 and not (set(syllable) & INVALID_LETTERS)


def _as_token(match):
    text = match.group().lower()
    return NUM_TOKEN if text.isdigit() else text


def iter_ngrams(text):
    """Yield (syllable, next_syllable_or_None) over word and number tokens.

    Digit runs are mapped to the <num> context token. Bigrams are only
    formed across pure-whitespace gaps, so punctuation between words does
    not create false collocations.
    """
    matches = list(TOKEN_RE.finditer(text))
    for i, match in enumerate(matches):
        token = _as_token(match)
        next_token = None
        if i + 1 < len(matches):
            gap = text[match.end():matches[i + 1].start()]
            if gap != "" and gap.isspace():
                next_token = _as_token(matches[i + 1])
        yield token, next_token


def count_text(text, weight, unigrams, bigrams):
    for token, next_token in iter_ngrams(text):
        if not valid_syllable(token):
            continue
        unigrams[token] += weight
        if next_token and valid_syllable(next_token):
            bigrams[(token, next_token)] += weight


def build():
    unigrams = Counter()
    bigrams = Counter()

    print("Counting VNTC train...")
    with open(VNTC_TRAIN, encoding="utf-8") as f:
        for line in f:
            # drop the "__label__xxx" prefix
            count_text(line.split("  ", 1)[-1], 1, unigrams, bigrams)

    print("Counting dictionary...")
    dictionary = Dictionary.Instance()
    for word, senses in dictionary.words.items():
        count_text(word, HEADWORD_WEIGHT, unigrams, bigrams)
        for sense in senses:
            if not isinstance(sense, dict):
                continue
            for field in ("definition", "example"):
                if sense.get(field):
                    count_text(sense[field], 1, unigrams, bigrams)

    pruned_unigrams = {k: v for k, v in unigrams.items() if v >= UNIGRAM_MIN_COUNT}
    pruned_bigrams = {
        k: v for k, v in bigrams.items()
        if v >= BIGRAM_MIN_COUNT and k[0] in pruned_unigrams and k[1] in pruned_unigrams
    }
    print(f"unigrams: {len(unigrams)} -> {len(pruned_unigrams)}")
    print(f"bigrams: {len(bigrams)} -> {len(pruned_bigrams)}")

    model = {"unigrams": pruned_unigrams, "bigrams": pruned_bigrams}
    with gzip.open(OUTPUT, "wb", compresslevel=9) as f:
        pickle.dump(model, f, protocol=4)
    print(f"Model written to {OUTPUT}")


if __name__ == "__main__":
    build()
