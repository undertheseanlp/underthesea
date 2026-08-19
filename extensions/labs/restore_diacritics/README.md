# Restore Diacritics Lab

Tooling for the `underthesea.pipeline.restore_diacritics` module
(GitHub issue [#766](https://github.com/undertheseanlp/underthesea/issues/766)).

## Method

Lightweight mode — no deep learning dependencies:

1. The input is split on whitespace and hard punctuation (`, ; ! ? …`),
   so words joined without spaces ("mai,Lan") are still separated.
   Chunks that are neither alphabetic nor numeric (URLs, emails, `9X`)
   are left untouched and break the context chain, as does punctuation.
   Numeric chunks are left untouched but stay in the chain as a `<num>`
   context token, so "30 thang 3" reads as a date.
2. For each alphabetic token, candidate accented forms are looked up by
   stripping diacritics from the model vocabulary. Tokens that already
   carry diacritics are kept as-is.
3. The best sequence of candidates is decoded with the Viterbi algorithm
   over an interpolated syllable bigram model
   (`P = 0.8 * P_bigram + 0.2 * P_unigram`, add-one smoothed).

## Model

`rd_model_YYYY_MM_DD.bin` is a gzipped pickle with pruned counts:

| Source | Weight |
|--------|--------|
| VNTC news corpus (train split, 33,759 articles) | 1 |
| Dictionary definitions and examples (31k entries) | 1 |
| Dictionary headwords | 3 |

Pruning: unigram count >= 5, bigram count >= 5
(~13.6k unigrams, ~225k bigrams, ~1.5 MB gzipped).

Rebuild with:

```bash
python -c "from underthesea.data_fetcher import DataFetcher; DataFetcher.download_data('VNTC', None)"
python extensions/labs/restore_diacritics/build_model.py
```

## Evaluation

Two test sets of 300 sentences each, evaluated on syllable accuracy over
alphabetic tokens (`evaluate.py` reproduces the VNTC rows):

* **VNTC-test** — sampled from the VNTC test split (news, in-domain,
  never used for training)
* **Wiki** — random Vietnamese Wikipedia article sentences fetched in
  August 2026 (out-of-domain for every system below)

| System | Size | VNTC-test | Wiki |
|--------|------|-----------|------|
| Unigram baseline (no context) | 1.5 MB | 71.9% | 73.4% |
| **restore_diacritics (this module)** | **1.5 MB** | **95.0%** | **91.9%** |
| [XLM-R large accent marker](https://huggingface.co/peterhung/vietnamese-accent-marker-xlm-roberta) | 2.2 GB | 92.1% | 89.6% |

The XLM-R model (self-reported 97% in-domain) was run with the author's
published decoding, patched to also apply tags to capitalized words.
On CPU it needs 140-180 ms per sentence; this module needs ~0.5-1 ms
(~200x faster) after a lazy one-off model load (~0.3 s).

## Error analysis notes

* Roughly two thirds of this module's errors pick a wrong variant of an
  in-vocabulary syllable (tone mark or letter shape); OOV syllables are
  a negligible cause. Context length dominates: isolated words fail
  4-5x more often than words inside a chain of 3+ syllables — which is
  why numbers act as `<num>` context instead of breaking chains.
* Residual weak spots: proper nouns that collide with common bigrams
  ("Hoàng Văn" vs "vẫn"), domain shift (Wikipedia species articles:
  "loài", "họ"), and genuinely semantic ambiguities ("hoà tẻ nhạt" vs
  "hòa tệ nhất") that also defeat the 2.2 GB transformer.
* The XLM-R baseline errs differently: it mostly fails to add any marks
  at all — it misses "đã" in 100% of occurrences on both test sets —
  and is 2x weaker on Vietnamese proper nouns. The two systems disagree
  enough that an oracle ensemble would reach ~97.7% / ~95.5%, which is
  the motivation for a future context-aware AI mode (the second mode
  suggested in #766).
* Mixing a Wikipedia corpus into the model build is the highest-value
  future improvement (23% of Wiki errors come from three news-biased
  confusions: "loài→loại", "họ→hồ", "tháng→thắng").
