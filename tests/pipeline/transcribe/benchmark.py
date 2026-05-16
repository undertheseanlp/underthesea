"""End-to-end accuracy benchmark for `underthesea.pipeline.transcribe`.

Usage
-----

1. Against a local manifest (TSV: "<audio_path>\\t<reference_text>")::

       python -m tests.pipeline.transcribe.benchmark \\
           --manifest path/to/manifest.tsv --model large

2. Against a Hugging Face audio dataset (needs network access to HF)::

       python -m tests.pipeline.transcribe.benchmark \\
           --hf-dataset doof-ferb/vlsp2020_vinai_100h --split test \\
           --audio-column audio --text-column transcription \\
           --num-samples 50 --model small

Outputs per-sample predictions and aggregate WER / CER computed with
``jiwer``. Reference text is lower-cased and stripped of punctuation
before comparison to match common ASR evaluation protocol.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path


def _strip_punct(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[\.,!?;:\"'()\[\]{}—–\-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _load_manifest(path: str, limit: int | None):
    pairs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        pairs.append((parts[0].strip(), parts[1].strip()))
        if limit and len(pairs) >= limit:
            break
    return pairs


def _load_hf(dataset: str, split: str, audio_col: str, text_col: str,
             limit: int | None, cache_dir: str | None):
    from datasets import Audio, load_dataset

    ds = load_dataset(dataset, split=split, cache_dir=cache_dir)
    ds = ds.cast_column(audio_col, Audio(sampling_rate=16_000))
    pairs = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        pairs.append((row[audio_col]["array"], row[text_col]))
    return pairs


def run(samples, model: str, num_beams: int, normalize: bool):
    from jiwer import cer, wer

    from underthesea.pipeline.transcribe import transcribe

    refs, hyps = [], []
    t0 = time.time()
    for i, (audio, ref) in enumerate(samples, 1):
        hyp = transcribe(audio, model=model, num_beams=num_beams,
                         normalize=normalize)
        ref_n, hyp_n = _strip_punct(ref), _strip_punct(hyp)
        refs.append(ref_n)
        hyps.append(hyp_n)
        print(f"[{i:>3}/{len(samples)}] REF: {ref_n}")
        print(f"        HYP: {hyp_n}")
    dt = time.time() - t0

    w = wer(refs, hyps) * 100
    c = cer(refs, hyps) * 100
    print()
    print(f"model         : {model}")
    print(f"samples       : {len(samples)}")
    print(f"elapsed       : {dt:.1f}s  ({dt / max(len(samples), 1):.2f}s / sample)")
    print(f"WER           : {w:.2f}%")
    print(f"CER           : {c:.2f}%")
    return w, c


def main(argv=None):
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest",
                     help="TSV file: '<audio_path>\\t<reference_text>' per line")
    src.add_argument("--hf-dataset", help="Hugging Face audio dataset id")
    p.add_argument("--split", default="test")
    p.add_argument("--audio-column", default="audio")
    p.add_argument("--text-column", default="transcription")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--num-samples", type=int, default=50)
    p.add_argument("--model", default="large",
                   help="Model alias or HF id (default: large = PhoWhisper-large)")
    p.add_argument("--num-beams", type=int, default=5)
    p.add_argument("--no-normalize", action="store_true")
    args = p.parse_args(argv)

    if args.manifest:
        samples = _load_manifest(args.manifest, args.num_samples)
    else:
        samples = _load_hf(args.hf_dataset, args.split, args.audio_column,
                           args.text_column, args.num_samples, args.cache_dir)

    if not samples:
        print("No samples found.", file=sys.stderr)
        return 1

    run(samples, model=args.model, num_beams=args.num_beams,
        normalize=not args.no_normalize)
    return 0


if __name__ == "__main__":
    sys.exit(main())
