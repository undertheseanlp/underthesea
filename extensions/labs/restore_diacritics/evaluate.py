"""Evaluate restore_diacritics on held-out sentences from the VNTC test split.

Usage (from repo root, after downloading VNTC — see build_model.py):
    python extensions/labs/restore_diacritics/evaluate.py
"""
import random
import re
import time
from os.path import expanduser

from underthesea.pipeline.restore_diacritics import restore_diacritics
from underthesea.pipeline.restore_diacritics.restorer import (
    WORD_RE,
    DiacriticsRestorer,
)
from underthesea.utils.vietnamese_features import remove_tone

VNTC_TEST = expanduser("~/.underthesea/datasets/VNTC/test.txt")

VN_RE = re.compile(r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợ"
                   r"ùúủũụưừứửữựỳýỷỹỵđ]")


def load_sentences(path, limit=300):
    """One clean sentence per sampled article: 6-60 tokens, mostly accented."""
    random.seed(42)
    with open(path, encoding="utf-8") as f:
        lines = [line for i, line in enumerate(f) if i % 37 == 0]
    random.shuffle(lines)
    sentences = []
    for line in lines:
        text = line.split("  ", 1)[-1].strip()
        for raw in re.split(r"(?<=[.!?])\s+", text):
            sentence = raw.strip()
            tokens = WORD_RE.findall(sentence)
            if len(tokens) < 6 or len(tokens) > 60:
                continue
            accented = sum(1 for t in tokens if VN_RE.search(t.lower()))
            if accented / len(tokens) < 0.5:
                continue
            sentences.append(sentence)
            break
        if len(sentences) >= limit:
            break
    return sentences


def unigram_baseline(text):
    """Most frequent variant per syllable, no context."""
    restorer = DiacriticsRestorer.Instance()

    def best(match):
        token = match.group()
        lower = token.lower()
        if lower != remove_tone(lower):
            return token
        variants = restorer.variants.get(lower)
        if not variants:
            return token
        chosen = max(variants, key=lambda v: restorer.unigrams.get(v, 0))
        if chosen == lower:
            return token
        if token.isupper() and len(token) > 1:
            return chosen.upper()
        if token[:1].isupper():
            return chosen[:1].upper() + chosen[1:]
        return chosen
    return WORD_RE.sub(best, text)


def evaluate(name, restore, sentences):
    syllables_correct = syllables_total = 0
    sentences_correct = sentences_total = 0
    start = time.time()
    for gold in sentences:
        restored = restore(remove_tone(gold))
        gold_tokens = [t.lower() for t in WORD_RE.findall(gold)]
        restored_tokens = [t.lower() for t in WORD_RE.findall(restored)]
        if len(gold_tokens) != len(restored_tokens):
            continue
        sentences_total += 1
        all_ok = True
        for g, p in zip(gold_tokens, restored_tokens):
            syllables_total += 1
            if g == p:
                syllables_correct += 1
            else:
                all_ok = False
        sentences_correct += all_ok
    elapsed = time.time() - start
    print(f"{name:20s} syllable_acc={syllables_correct / syllables_total:.4f} "
          f"sentence_acc={sentences_correct / sentences_total:.4f} "
          f"({syllables_total} syllables, "
          f"{elapsed / len(sentences) * 1000:.1f} ms/sentence)")


if __name__ == "__main__":
    sentences = load_sentences(VNTC_TEST)
    print(f"{len(sentences)} sentences from VNTC test split")
    evaluate("unigram baseline", unigram_baseline, sentences)
    evaluate("restore_diacritics", restore_diacritics, sentences)
