# flake8: noqa

from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import librosa
from einops import rearrange
from jax.numpy import ndarray


def rolling_window(a: ndarray, window: int, hop_length: int):
    """return a stack of overlap subsequence of an array.
    ``return jnp.stack( [a[0:10], a[5:15], a[10:20],...], axis=0)``
    Source: https://github.com/google/jax/issues/3171
    Args:
      a (ndarray): input array of shape `[L, ...]`
      window (int): length of each subarray (window).
      hop_length (int): distance between neighbouring windows.
    """

    idx = (
        jnp.arange(window)[:, None] +
        jnp.arange((len(a) - window) // hop_length + 1)[None, :] * hop_length
    )
    return a[idx]


@partial(jax.jit, static_argnums=[1, 2, 3, 4, 5, 6])
def stft(
    y: ndarray,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: str = "hann",
    center: bool = True,
    pad_mode: str = "reflect",
):
    """A jax reimplementation of ``librosa.stft`` function."""

    if win_length is None:
        win_length = n_fft

    if hop_length is None:
        hop_length = win_length // 4

    if window == "hann":
        fft_window = jnp.hanning(win_length + 1)[:-1]
    else:
        raise RuntimeError(f"{window} window function is not supported!")

    pad_len = (n_fft - win_length) // 2
    fft_window = jnp.pad(fft_window, (pad_len, pad_len), mode="constant")
    fft_window = fft_window[:, None]
    if center:
        y = jnp.pad(y, int(n_fft // 2), mode=pad_mode)

    # jax does not support ``np.lib.stride_tricks.as_strided`` function
    # see https://github.com/google/jax/issues/3171 for comments.
    y_frames = rolling_window(y, n_fft, hop_length) * fft_window
    stft_matrix = jnp.fft.fft(y_frames, axis=0)
    d = int(1 + n_fft // 2)
    return stft_matrix[:d]


@partial(jax.jit, static_argnums=[1, 2, 3, 4, 5, 6])
def batched_stft(
    y: ndarray,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: str,
    center: bool = True,
    pad_mode: str = "reflect",
):
    """Batched version of ``stft`` function.
    TN => FTN
    """

    assert len(y.shape) >= 2
    if window == "hann":
        fft_window = jnp.hanning(win_length + 1)[:-1]
    else:
        raise RuntimeError(f"{window} window function is not supported!")
    pad_len = (n_fft - win_length) // 2
    if pad_len > 0:
        fft_window = jnp.pad(fft_window, (pad_len, pad_len), mode="constant")
        win_length = n_fft
    else:
        fft_window = fft_window
    if center:
        pad_width = ((n_fft // 2, n_fft // 2),) + ((0, 0),) * (len(y.shape) - 1)
        y = jnp.pad(y, pad_width, mode=pad_mode)

    # jax does not support ``np.lib.stride_tricks.as_strided`` function
    # see https://github.com/google/jax/issues/3171 for comments.
    y_frames = rolling_window(y, n_fft, hop_length)
    fft_window = jnp.reshape(fft_window, (-1,) + (1,) * (len(y.shape)))
    y_frames = y_frames * fft_window
    stft_matrix = jnp.fft.fft(y_frames, axis=0)
    d = int(1 + n_fft // 2)
    return stft_matrix[:d]


class MelFilter:
    """Convert waveform to mel spectrogram."""

    def __init__(self, sample_rate: int, n_fft: int, n_mels: int, fmin=0.0, fmax=8000):
        self.melfb = jax.device_put(
            librosa.filters.mel(
                sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
            )
        )
        self.n_fft = n_fft

    def __call__(self, y: ndarray) -> ndarray:
        hop_length = self.n_fft // 4
        window_length = self.n_fft
        assert len(y.shape) == 2
        y = rearrange(y, "n s -> s n")
        p = (self.n_fft - hop_length) // 2
        y = jnp.pad(y, ((p, p), (0, 0)), mode="reflect")
        spec = batched_stft(
            y, self.n_fft, hop_length, window_length, "hann", False, "reflect"
        )
        mag = jnp.sqrt(jnp.square(spec.real) + jnp.square(spec.imag) + 1e-9)
        mel = jnp.einsum("ms,sfn->nfm", self.melfb, mag)
        cond = jnp.log(jnp.clip(mel, a_min=1e-5, a_max=None))
        return cond
