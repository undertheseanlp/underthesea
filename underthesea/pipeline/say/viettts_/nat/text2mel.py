# flake8: noqa

import pickle
from argparse import ArgumentParser
from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from .config import FLAGS, DurationInput
from .data_loader import load_phonemes_set
from .model import AcousticModel, DurationModel


def load_lexicon(fn):
    lines = open(fn, "r").readlines()
    lines = [l.lower().strip().split("\t") for l in lines]
    return dict(lines)


def predict_duration(tokens, ckpt_file):
    def fwd_(x):
        return DurationModel(is_training=False)(x)

    forward_fn = jax.jit(hk.transform_with_state(fwd_).apply)
    with open(ckpt_file, "rb") as f:
        dic = pickle.load(f)
    x = DurationInput(
        np.array(tokens, dtype=np.int32)[None, :],
        np.array([len(tokens)], dtype=np.int32),
        None,
    )
    return forward_fn(dic["params"], dic["aux"], dic["rng"], x)[0]


def text2tokens(text, lexicon_fn):
    phonemes = load_phonemes_set()
    lexicon = load_lexicon(lexicon_fn)

    words = text.strip().lower().split()
    tokens = [FLAGS.sil_index]
    for word in words:
        if word in FLAGS.special_phonemes:
            tokens.append(phonemes.index(word))
        elif word in lexicon:
            p = lexicon[word]
            p = p.split()
            p = [phonemes.index(pp) for pp in p]
            tokens.extend(p)
            tokens.append(FLAGS.word_end_index)
        else:
            for p in word:
                if p in phonemes:
                    tokens.append(phonemes.index(p))
            tokens.append(FLAGS.word_end_index)
    tokens.append(FLAGS.sil_index)  # silence
    return tokens

dic = None
def predict_mel(tokens, durations, ckpt_fn):
    global dic
    with open(ckpt_fn, "rb") as f:
        if dic is None:
            dic = pickle.load(f)
        last_step, params, aux, rng, optim_state = (
            dic["step"],
            dic["params"],
            dic["aux"],
            dic["rng"],
            dic["optim_state"],
        )

    @hk.transform_with_state
    def forward(tokens, durations, n_frames):
        net = AcousticModel(is_training=False)
        return net.inference(tokens, durations, n_frames)

    durations = durations * FLAGS.sample_rate / (FLAGS.n_fft // 4)
    n_frames = int(jnp.sum(durations).item())
    predict_fn = jax.jit(forward.apply, static_argnums=[5])
    tokens = np.array(tokens, dtype=np.int32)[None, :]
    return predict_fn(params, aux, rng, tokens, durations, n_frames)[0]


def text2mel(
    text: str,
    lexicon_fn=FLAGS.data_dir / "lexicon.txt",
    silence_duration: float = -1.0,
    acoustic_ckpt=FLAGS.ckpt_dir / "acoustic_latest_ckpt.pickle",
    duration_ckpt=FLAGS.ckpt_dir / "duration_latest_ckpt.pickle",
):
    tokens = text2tokens(text, lexicon_fn)
    durations = predict_duration(tokens, duration_ckpt)
    durations = jnp.where(
        np.array(tokens)[None, :] == FLAGS.sil_index,
        jnp.clip(durations, a_min=silence_duration, a_max=None),
        durations,
    )
    durations = jnp.where(
        np.array(tokens)[None, :] == FLAGS.word_end_index, 0.0, durations
    )
    mels = predict_mel(tokens, durations, acoustic_ckpt)
    if tokens[-1] == FLAGS.sil_index:
        end_silence = durations[0, -1].item()
        silence_frame = int(end_silence * FLAGS.sample_rate / (FLAGS.n_fft // 4))
        mels = mels[:, : (mels.shape[1] - silence_frame)]
    return mels


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mel = text2mel(args.text)
    plt.figure(figsize=(10, 5))
    plt.imshow(mel[0].T, origin="lower", aspect="auto")
    plt.savefig(str(args.output))
    plt.close()
    mel = jax.device_get(mel)
    mel.tofile("clip.mel")
