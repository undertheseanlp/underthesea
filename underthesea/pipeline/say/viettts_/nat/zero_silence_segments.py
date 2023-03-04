# flake8: noqa

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from textgrid import TextGrid
from tqdm.auto import tqdm

from .config import FLAGS

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("-o", "--output-dir", type=Path, required=True)
    args = parser.parse_args()

    files = sorted(FLAGS.data_dir.glob("*.TextGrid"))
    for fn in tqdm(files):
        tg = TextGrid.fromFile(str(fn.resolve()))
        wav_fn = FLAGS.data_dir / f"{fn.stem}.wav"
        sr, y = wavfile.read(wav_fn)
        y = np.copy(y)
        for phone in tg[1]:
            if phone.mark in FLAGS.special_phonemes:
                l = int(phone.minTime * sr)
                r = int(phone.maxTime * sr)
                y[l:r] = 0
        out_file = args.output_dir / f"{fn.stem}.wav"
        wavfile.write(out_file, sr, y)
